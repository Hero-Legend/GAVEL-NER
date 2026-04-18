import datetime
import json
import os
import random
from dataclasses import dataclass

import jieba.posseg as pseg
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from seqeval.metrics import classification_report


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

MODEL_PATH = os.environ.get("CAIL_MODEL_PATH", "./model_path/chinese-roberta-wwm-ext")
DATA_PATH = os.environ.get("CAIL_DATA_PATH", "./data/formatted_data_fixed.json")
RUN_TAG = os.environ.get("CAIL_RUN_TAG", "cail_learning_curve")
MAX_LENGTH = int(os.environ.get("CAIL_MAX_LENGTH", "256"))
EPOCHS = float(os.environ.get("CAIL_EPOCHS", "8"))
TRAIN_BATCH = int(os.environ.get("CAIL_BATCH", "8"))
EVAL_BATCH = int(os.environ.get("CAIL_EVAL_BATCH", "8"))
GRAD_ACC = int(os.environ.get("CAIL_GRAD_ACC", "2"))
LEARNING_RATE = float(os.environ.get("CAIL_LR", "2e-5"))
WARMUP_RATIO = float(os.environ.get("CAIL_WARMUP_RATIO", "0.1"))
WEIGHT_DECAY = float(os.environ.get("CAIL_WEIGHT_DECAY", "0.01"))
RATIOS = [float(x) for x in os.environ.get("CAIL_RATIOS", "0.1,0.3,0.5,0.7,1.0").split(",") if x.strip()]

bmes2id = {"[PAD]": 0, "B": 1, "M": 2, "E": 3, "S": 4}
pos2id = {"[PAD]": 0}


def parse_span(span):
    if isinstance(span, (list, tuple)) and len(span) == 1 and isinstance(span[0], str):
        parts = span[0].replace(";", ",").split(",")
        return int(parts[0]), int(parts[1])
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        return int(span[0]), int(span[1])
    if isinstance(span, str):
        parts = span.replace(";", ",").split(",")
        return int(parts[0]), int(parts[1])
    raise ValueError("bad span")


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

    if len(bmes_tags) != len(text) or len(pos_tags) != len(text):
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
                    start, end = parse_span(span)
                except Exception:
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
    def __init__(self, model_path, num_labels, use_external=True, num_lexical=5, num_structural=100):
        super().__init__()
        self.use_external = use_external
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

        if self.use_external:
            lex_embeds = self.lexical_embedding(lexical_ids)
            struct_embeds = self.structural_embedding(structural_ids)

            g_lex = torch.sigmoid(self.gate_lex(torch.cat([inputs_embeds, lex_embeds], dim=-1)))
            fused_1 = inputs_embeds + g_lex * lex_embeds

            g_struct = torch.sigmoid(self.gate_struct(torch.cat([fused_1, struct_embeds], dim=-1)))
            fused_embeds = fused_1 + g_struct * struct_embeds
        else:
            fused_embeds = inputs_embeds

        fused_embeds = self.dropout(fused_embeds)
        return self.bert_for_token_cls(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=self.training,
        )


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


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


class CustomTrainer(Trainer):
    def __init__(self, use_contrastive=False, use_fgm=False, contrastive_weight=0.1, o_label_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_contrastive = use_contrastive
        self.use_fgm = use_fgm
        self.contrastive_weight = contrastive_weight
        self.o_label_id = o_label_id
        self.contrastive_loss_fn = SupConLoss()
        self.fgm = FGM(self.model) if use_fgm else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss

        if self.model.training and self.use_contrastive:
            hidden_states = outputs.hidden_states[-1]
            contrastive_loss = self.contrastive_loss_fn(hidden_states, labels, o_label_id=self.o_label_id)
        else:
            contrastive_loss = torch.tensor(0.0, device=loss.device)

        total_loss = loss + self.contrastive_weight * contrastive_loss
        return (total_loss, outputs) if return_outputs else total_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()

        if self.use_fgm and self.fgm is not None:
            self.fgm.attack()
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            loss_adv.backward()
            self.fgm.restore()

        return loss.detach()


@dataclass
class ExpConfig:
    name: str
    use_external: bool
    use_contrastive: bool
    use_fgm: bool
    contrastive_weight: float


def compute_metrics_builder(id2label):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(np.array(logits), axis=-1)
        labels_np = np.array(labels)

        true_labels = [[id2label[l] for l in label if l != -100] for label in labels_np]
        true_predictions = [
            [id2label[p] for p, l in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels_np)
        ]

        results = classification_report(true_labels, true_predictions, output_dict=True)
        return {
            "precision": results["macro avg"]["precision"],
            "recall": results["macro avg"]["recall"],
            "f1": results["macro avg"]["f1-score"],
        }

    return compute_metrics


def build_splits(n_items, seed=42):
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_size = int(0.8 * n_items)
    val_size = int(0.1 * n_items)
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]
    return train_idx, val_idx, test_idx


def subset_by_indices(items, indices):
    return [items[i] for i in indices]


def build_ratio_subset(indices, ratio, seed=42):
    if ratio >= 1.0:
        return list(indices)
    n_keep = max(1, int(len(indices) * ratio))
    rng = random.Random(seed + int(ratio * 1000))
    chosen = list(indices)
    rng.shuffle(chosen)
    return chosen[:n_keep]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CAIL data file missing: {DATA_PATH}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    sentences, labels, lexical_feats, structural_feats = load_data(DATA_PATH)

    unique_labels = sorted(set(tag for doc in labels for tag in doc))
    label2id = {tag: i for i, tag in enumerate(unique_labels)}
    id2label = {i: tag for tag, i in label2id.items()}

    train_idx, val_idx, test_idx = build_splits(len(sentences), seed=42)
    base_train_size = len(train_idx)

    exp_configs = [
        ExpConfig(name="roberta_baseline", use_external=False, use_contrastive=False, use_fgm=False, contrastive_weight=0.0),
        ExpConfig(name="mvcl_full", use_external=True, use_contrastive=True, use_fgm=True, contrastive_weight=0.1),
    ]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"./outputs/{RUN_TAG}_{timestamp}"
    os.makedirs(output_root, exist_ok=True)
    all_results = []

    for ratio in RATIOS:
        ratio_train_idx = build_ratio_subset(train_idx, ratio, seed=42)

        tr_sent = subset_by_indices(sentences, ratio_train_idx)
        tr_lab = subset_by_indices(labels, ratio_train_idx)
        tr_lex = subset_by_indices(lexical_feats, ratio_train_idx)
        tr_st = subset_by_indices(structural_feats, ratio_train_idx)

        va_sent = subset_by_indices(sentences, val_idx)
        va_lab = subset_by_indices(labels, val_idx)
        va_lex = subset_by_indices(lexical_feats, val_idx)
        va_st = subset_by_indices(structural_feats, val_idx)

        te_sent = subset_by_indices(sentences, test_idx)
        te_lab = subset_by_indices(labels, test_idx)
        te_lex = subset_by_indices(lexical_feats, test_idx)
        te_st = subset_by_indices(structural_feats, test_idx)

        ratio_result = {
            "ratio": ratio,
            "train_samples": len(ratio_train_idx),
            "validation_samples": len(val_idx),
            "test_samples": len(test_idx),
            "results": [],
        }

        for cfg in exp_configs:
            print(f"\n=== Running ratio={ratio:.2f}, config={cfg.name} ===")

            train_ds = NERDataset(tr_sent, tr_lab, tr_lex, tr_st, tokenizer, label2id, max_length=MAX_LENGTH)
            val_ds = NERDataset(va_sent, va_lab, va_lex, va_st, tokenizer, label2id, max_length=MAX_LENGTH)
            test_ds = NERDataset(te_sent, te_lab, te_lex, te_st, tokenizer, label2id, max_length=MAX_LENGTH)

            model = MVCLBert(
                model_path=MODEL_PATH,
                num_labels=len(label2id),
                use_external=cfg.use_external,
                num_structural=max(100, len(pos2id)),
            )

            safe_ratio = str(ratio).replace(".", "p")
            out_dir = os.path.join(output_root, f"ratio_{safe_ratio}", cfg.name)
            os.makedirs(out_dir, exist_ok=True)

            args = TrainingArguments(
                output_dir=out_dir,
                eval_strategy="epoch",
                save_strategy="no",
                logging_strategy="steps",
                logging_steps=100,
                per_device_train_batch_size=TRAIN_BATCH,
                per_device_eval_batch_size=EVAL_BATCH,
                gradient_accumulation_steps=GRAD_ACC,
                num_train_epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                warmup_ratio=WARMUP_RATIO,
                weight_decay=WEIGHT_DECAY,
                eval_accumulation_steps=8,
                remove_unused_columns=False,
                report_to="none",
                fp16=False,
            )

            trainer = CustomTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                compute_metrics=compute_metrics_builder(id2label),
                use_contrastive=cfg.use_contrastive,
                use_fgm=cfg.use_fgm,
                contrastive_weight=cfg.contrastive_weight,
                o_label_id=label2id.get("O", None),
            )

            trainer.train()
            metrics = trainer.evaluate(test_ds)
            metrics["config"] = cfg.name
            metrics["ratio"] = ratio
            metrics["train_samples"] = len(ratio_train_idx)
            metrics["use_external"] = cfg.use_external
            metrics["use_contrastive"] = cfg.use_contrastive
            metrics["use_fgm"] = cfg.use_fgm
            ratio_result["results"].append(metrics)

            with open(os.path.join(out_dir, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")

            del trainer, model
            torch.cuda.empty_cache()

        baseline = next(x for x in ratio_result["results"] if x["config"] == "roberta_baseline")
        full = next(x for x in ratio_result["results"] if x["config"] == "mvcl_full")
        ratio_result["delta_f1_full_minus_baseline"] = full.get("eval_f1", 0.0) - baseline.get("eval_f1", 0.0)
        all_results.append(ratio_result)

    summary = {
        "timestamp": timestamp,
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "ratios": RATIOS,
        "train_pool_size": base_train_size,
        "validation_size": len(val_idx),
        "test_size": len(test_idx),
        "results": all_results,
    }

    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# CAIL Learning Curve\n\n")
        f.write(f"- Data file: {DATA_PATH}\n")
        f.write(f"- Model path: {MODEL_PATH}\n")
        f.write(f"- Training ratios: {', '.join(f'{r:.1f}' for r in RATIOS)}\n")
        f.write(f"- Training pool size: {base_train_size}\n")
        f.write(f"- Validation size: {len(val_idx)}\n")
        f.write(f"- Test size: {len(test_idx)}\n\n")
        f.write("| Train Ratio | Train Samples | Baseline F1 | GAVEL-Full F1 | Delta |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for row in all_results:
            baseline = next(x for x in row["results"] if x["config"] == "roberta_baseline")
            full = next(x for x in row["results"] if x["config"] == "mvcl_full")
            f.write(
                f"| {row['ratio']:.1f} | {row['train_samples']} | {baseline.get('eval_f1', 0):.4f} | {full.get('eval_f1', 0):.4f} | {row['delta_f1_full_minus_baseline']:.4f} |\n"
            )

    print(f"\nDone. Summary saved to: {output_root}/summary.md")


if __name__ == "__main__":
    main()
