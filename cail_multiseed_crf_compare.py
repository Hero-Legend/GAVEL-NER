import datetime
import json
import os
import random
from dataclasses import dataclass
from statistics import mean, pstdev

import jieba.posseg as pseg
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split
from torchcrf import CRF
from transformers import BertForTokenClassification, BertModel, BertTokenizerFast, Trainer, TrainingArguments
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import classification_report


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


bmes2id = {"[PAD]": 0, "B": 1, "M": 2, "E": 3, "S": 4}
pos2id = {"[PAD]": 0}
MODEL_PATH = "./model_path/chinese-roberta-wwm-ext"
DATA_PATH = "./data/formatted_data_fixed.json"
MAX_LENGTH = 256


def get_lexical_structural_features(text):
    bmes_tags, pos_tags = [], []
    for word, flag in pseg.cut(text):
        length = len(word)
        if length == 1:
            bmes_tags.append("S")
        else:
            bmes_tags.extend(["B"] + ["M"] * (length - 2) + ["E"])
        if flag not in pos2id:
            pos2id[flag] = len(pos2id)
        pos_tags.extend([flag] * length)

    if len(bmes_tags) != len(text):
        bmes_tags = ["S"] * len(text)
        if "x" not in pos2id:
            pos2id["x"] = len(pos2id)
        pos_tags = ["x"] * len(text)

    return [bmes2id[t] for t in bmes_tags], [pos2id[t] for t in pos_tags]


def load_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences, labels, lexical_features, structural_features = [], [], [], []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if not text:
            continue

        label_seq = ["O"] * len(text)
        for entity in item.get("entities", []):
            entity_type = entity.get("label") or entity.get("type") or "UNKNOWN"
            raw_spans = entity.get("span", [])
            if len(raw_spans) > 0 and not isinstance(raw_spans[0], list):
                raw_spans = [raw_spans]

            for span in raw_spans:
                try:
                    if isinstance(span, (list, tuple)) and len(span) == 1 and isinstance(span[0], str):
                        parts = span[0].replace(";", ",").split(",")
                        start, end = int(parts[0]), int(parts[1])
                    elif isinstance(span, (list, tuple)) and len(span) >= 2:
                        start, end = int(span[0]), int(span[1])
                    elif isinstance(span, str):
                        parts = span.replace(";", ",").split(",")
                        start, end = int(parts[0]), int(parts[1])
                    else:
                        continue
                except (ValueError, TypeError, IndexError):
                    continue

                if start >= len(label_seq) or end > len(label_seq) or start >= end:
                    continue

                label_seq[start] = f"B-{entity_type}"
                for i in range(start + 1, end):
                    label_seq[i] = f"I-{entity_type}"

        lex, struct = get_lexical_structural_features(text)
        if len(lex) != len(text):
            continue

        sentences.append(list(text))
        labels.append(label_seq)
        lexical_features.append(lex)
        structural_features.append(struct)

    return sentences, labels, lexical_features, structural_features


class NERDataset(Dataset):
    def __init__(self, sentences, labels, lex_feats, struct_feats, tokenizer, label2id, max_length=256):
        self.sentences = sentences
        self.labels = labels
        self.lex_feats = lex_feats
        self.struct_feats = struct_feats
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = self.sentences[idx], self.labels[idx]
        lex_seq, struct_seq = self.lex_feats[idx], self.struct_feats[idx]
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        label_ids, lexical_ids, structural_ids = [], [], []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
                lexical_ids.append(0)
                structural_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(self.label2id[tags[word_idx]])
                lexical_ids.append(lex_seq[word_idx])
                structural_ids.append(struct_seq[word_idx])
            else:
                label_ids.append(-100)
                lexical_ids.append(0)
                structural_ids.append(0)
            previous_word_idx = word_idx

        encoding["labels"] = torch.tensor(label_ids)
        encoding["lexical_ids"] = torch.tensor(lexical_ids)
        encoding["structural_ids"] = torch.tensor(structural_ids)
        return {k: v.squeeze(0) for k, v in encoding.items()}


class MVCLBert(nn.Module):
    def __init__(self, model_path, num_labels, num_lexical=5, num_structural=100):
        super().__init__()
        self.bert_for_token_cls = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
        hidden_size = self.bert_for_token_cls.config.hidden_size
        self.lexical_embedding = nn.Embedding(num_lexical, hidden_size)
        self.structural_embedding = nn.Embedding(num_structural, hidden_size)
        self.gate_lex = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_struct = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)

        nn.init.zeros_(self.lexical_embedding.weight)
        nn.init.zeros_(self.structural_embedding.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, lexical_ids=None, structural_ids=None, labels=None):
        inputs_embeds = self.bert_for_token_cls.bert.embeddings.word_embeddings(input_ids)
        lex_embeds = self.lexical_embedding(lexical_ids)
        struct_embeds = self.structural_embedding(structural_ids)

        g_lex = torch.sigmoid(self.gate_lex(torch.cat([inputs_embeds, lex_embeds], dim=-1)))
        fused_1 = inputs_embeds + g_lex * lex_embeds
        g_struct = torch.sigmoid(self.gate_struct(torch.cat([fused_1, struct_embeds], dim=-1)))
        fused_embeds = fused_1 + g_struct * struct_embeds
        fused_embeds = self.dropout(fused_embeds)

        return self.bert_for_token_cls(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=self.training,
        )


class MVCLBertCrf(nn.Module):
    def __init__(self, model_path, num_labels, num_lexical=5, num_structural=100):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size
        self.lexical_embedding = nn.Embedding(num_lexical, hidden_size)
        self.structural_embedding = nn.Embedding(num_structural, hidden_size)
        self.gate_lex = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_struct = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.3)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

        nn.init.zeros_(self.lexical_embedding.weight)
        nn.init.zeros_(self.structural_embedding.weight)

    def forward(self, input_ids, attention_mask, token_type_ids=None, lexical_ids=None, structural_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state
        lex_embeds = self.lexical_embedding(lexical_ids)
        struct_embeds = self.structural_embedding(structural_ids)

        g_lex = torch.sigmoid(self.gate_lex(torch.cat([sequence_output, lex_embeds], dim=-1)))
        fused_1 = sequence_output + g_lex * lex_embeds
        g_struct = torch.sigmoid(self.gate_struct(torch.cat([fused_1, struct_embeds], dim=-1)))
        fused_embeds = fused_1 + g_struct * struct_embeds
        fused_embeds = self.dropout(fused_embeds)
        logits = self.classifier(fused_embeds)

        loss = None
        mask = attention_mask.bool()
        if labels is not None:
            crf_labels = labels.clone()
            crf_labels = torch.where(crf_labels == -100, torch.tensor(0, device=crf_labels.device), crf_labels)
            loss = -self.crf(logits, crf_labels, mask=mask, reduction="token_mean")

        best_paths = self.crf.decode(logits, mask=mask)
        fake_logits = torch.zeros_like(logits)
        for i, path in enumerate(best_paths):
            for j, tag in enumerate(path):
                fake_logits[i, j, tag] = 1.0

        return TokenClassifierOutput(
            loss=loss,
            logits=fake_logits,
            hidden_states=outputs.hidden_states if self.training else None,
        )


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, o_label_id=None):
        _, _, hidden_size = features.shape
        features = features.view(-1, hidden_size)
        labels = labels.view(-1)
        features = F.normalize(features, dim=-1)

        valid_mask = labels != -100
        if o_label_id is not None:
            valid_mask = valid_mask & (labels != o_label_id)

        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)

        features = features[valid_mask]
        labels = labels[valid_mask]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(label_mask.size(0), dtype=torch.bool, device=label_mask.device)
        label_mask = label_mask & (~eye)
        neg_mask = (~label_mask) & (~eye)

        pos_samples = similarity_matrix[label_mask].mean() if label_mask.any() else torch.tensor(0.0, device=features.device)
        neg_samples = similarity_matrix[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0, device=features.device)
        return -torch.log(torch.exp(pos_samples) / (torch.exp(pos_samples) + torch.exp(neg_samples) + 1e-8))


class ContrastiveTrainer(Trainer):
    def __init__(self, contrastive_weight=0.1, o_label_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.o_label_id = o_label_id
        self.contrastive_loss_fn = SupConLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        if model.training and outputs.hidden_states is not None:
            contrastive_loss = self.contrastive_loss_fn(outputs.hidden_states[-1], labels, o_label_id=self.o_label_id)
        else:
            contrastive_loss = torch.tensor(0.0, device=loss.device)
        total_loss = loss + self.contrastive_weight * contrastive_loss
        return (total_loss, outputs) if return_outputs else total_loss


class ContrastiveTrainerCrf(ContrastiveTrainer):
    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        bert_params = [(n, p) for n, p in self.model.named_parameters() if "bert" in n]
        other_params = [(n, p) for n, p in self.model.named_parameters() if "bert" not in n]

        grouped = [
            {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
            {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": self.args.learning_rate},
            {"params": [p for n, p in other_params if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay, "lr": 1e-3},
            {"params": [p for n, p in other_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": 1e-3},
        ]
        self.optimizer = torch.optim.AdamW(grouped)
        return self.optimizer


@dataclass
class ModelConfig:
    name: str
    use_crf: bool


def build_splits(seed, dataset):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    split_gen = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=split_gen)


def compute_metrics_builder(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(np.array(logits), axis=-1)
        labels_np = np.array(labels)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels_np]
        true_predictions = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels_np)]
        results = classification_report(true_labels, true_predictions, output_dict=True)
        return {
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
        }

    return compute_metrics


def summarize_runs(runs, key):
    vals = [r[key] for r in runs]
    return {"mean": mean(vals), "std": pstdev(vals) if len(vals) > 1 else 0.0}


def main():
    seeds = [42, 43, 44]
    os.makedirs("./outputs", exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

    sentences, labels, lexical_feats, structural_feats = load_data(DATA_PATH)
    unique_labels = sorted(set(tag for doc in labels for tag in doc))
    label2id = {tag: i for i, tag in enumerate(unique_labels)}
    id2label = {i: tag for tag, i in label2id.items()}
    dataset = NERDataset(sentences, labels, lexical_feats, structural_feats, tokenizer, label2id, max_length=MAX_LENGTH)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"./outputs/cail_multiseed_crf_compare_{timestamp}"
    os.makedirs(output_root, exist_ok=True)

    configs = [
        ModelConfig(name="mvcl_gated_crf_free", use_crf=False),
        ModelConfig(name="mvcl_with_crf", use_crf=True),
    ]

    all_results = []
    for cfg in configs:
        cfg_runs = []
        for seed in seeds:
            set_seed(seed)
            train_dataset, val_dataset, test_dataset = build_splits(seed, dataset)
            if cfg.use_crf:
                model = MVCLBertCrf(MODEL_PATH, len(label2id), num_structural=max(100, len(pos2id)))
                trainer_cls = ContrastiveTrainerCrf
            else:
                model = MVCLBert(MODEL_PATH, len(label2id), num_structural=max(100, len(pos2id)))
                trainer_cls = ContrastiveTrainer

            run_dir = os.path.join(output_root, cfg.name, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            training_args = TrainingArguments(
                output_dir=run_dir,
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="steps",
                logging_steps=100,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                gradient_accumulation_steps=2,
                num_train_epochs=12,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                eval_accumulation_steps=8,
                remove_unused_columns=False,
                report_to="none",
                fp16=False,
            )

            trainer = trainer_cls(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                processing_class=tokenizer,
                compute_metrics=compute_metrics_builder(id2label),
                contrastive_weight=0.1,
                o_label_id=label2id.get("O"),
            )

            trainer.train()
            metrics = trainer.evaluate(test_dataset)
            metrics["seed"] = seed
            metrics["config"] = cfg.name
            cfg_runs.append(metrics)

            with open(os.path.join(run_dir, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

            del trainer, model
            torch.cuda.empty_cache()

        all_results.append(
            {
                "config": cfg.name,
                "runs": cfg_runs,
                "eval_precision": summarize_runs(cfg_runs, "eval_precision"),
                "eval_recall": summarize_runs(cfg_runs, "eval_recall"),
                "eval_f1": summarize_runs(cfg_runs, "eval_f1"),
            }
        )

    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds, "results": all_results}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# CAIL Multi-seed CRF Comparison\n\n")
        f.write(f"- Seeds: {', '.join(str(x) for x in seeds)}\n\n")
        f.write("| Model | F1 mean | F1 std | Precision mean | Recall mean |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for row in all_results:
            f1 = row["eval_f1"]
            p = row["eval_precision"]
            r = row["eval_recall"]
            f.write(f"| {row['config']} | {f1['mean']:.4f} | {f1['std']:.4f} | {p['mean']:.4f} | {r['mean']:.4f} |\n")


if __name__ == "__main__":
    main()
