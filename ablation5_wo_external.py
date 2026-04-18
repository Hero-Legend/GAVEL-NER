import os
import json
import random
import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, random_split
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

bmes2id = {"[PAD]": 0}
pos2id = {"[PAD]": 0}


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

        # No external POS/BMES: all placeholders.
        lexical_features.append([0] * len(text))
        structural_features.append([0] * len(text))
        sentences.append(list(text))
        labels.append(label_seq)

    return sentences, labels, lexical_features, structural_features


DATA_PATH = "./data/formatted_data_fixed.json"
sentences, labels, lexical_feats, structural_feats = load_data(DATA_PATH)

unique_labels = set(tag for doc in labels for tag in doc)
label2id = {tag: i for i, tag in enumerate(sorted(unique_labels))}
id2label = {i: tag for tag, i in label2id.items()}

model_path = "./model_path/chinese-roberta-wwm-ext"
tokenizer = BertTokenizerFast.from_pretrained(model_path)


class MVCLDataset(Dataset):
    def __init__(self, sentences, labels, lex_feats, struct_feats, tokenizer, label2id, max_length=256):
        self.sentences = sentences
        self.labels = labels
        self.lex_feats = lex_feats
        self.struct_feats = struct_feats
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

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

    def __len__(self):
        return len(self.sentences)


dataset = MVCLDataset(sentences, labels, lexical_feats, structural_feats, tokenizer, label2id)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


class MVCL_BERT(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.bert_for_token_cls = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids, lexical_ids, structural_ids, labels=None):
        inputs_embeds = self.bert_for_token_cls.bert.embeddings.word_embeddings(input_ids)
        fused_embeds = self.dropout(inputs_embeds)
        return self.bert_for_token_cls(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
        )


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        _, _, hidden_size = features.shape
        features = features.view(-1, hidden_size)
        labels = labels.view(-1)

        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        valid_mask = (labels != -100).view(-1)
        valid_matrix_mask = valid_mask.unsqueeze(0) * valid_mask.unsqueeze(1)
        label_mask = label_mask & valid_matrix_mask

        pos_samples = similarity_matrix[label_mask].mean() if label_mask.any() else torch.tensor(0.0, device=features.device)
        neg_samples = similarity_matrix[~label_mask].mean() if (~label_mask).any() else torch.tensor(0.0, device=features.device)
        loss = -torch.log(torch.exp(pos_samples) / (torch.exp(pos_samples) + torch.exp(neg_samples) + 1e-8))
        return loss


class ContrastiveNERTrainer(Trainer):
    def __init__(self, contrastive_weight=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_loss_fn = SupConLoss()
        self.contrastive_weight = contrastive_weight
        self.fgm = FGM(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss
        if self.model.training:
            hidden_states = outputs.hidden_states[-1]
            contrastive_loss = self.contrastive_loss_fn(hidden_states, labels)
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

        self.fgm.attack()
        with self.compute_loss_context_manager():
            loss_adv = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss_adv = loss_adv / self.args.gradient_accumulation_steps
        loss_adv.backward()
        self.fgm.restore()

        return loss.detach()

    def _save_checkpoint(self, model, trial):
        # Disable checkpoint saving to avoid safetensors non-contiguous tensor errors.
        return

    def save_model(self, output_dir=None, _internal_call=False):
        return


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(np.array(logits), axis=-1)
    labels = np.array(labels)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]

    results = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1-score"],
    }


model = MVCL_BERT(model_path=model_path, num_labels=len(label2id))
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./outputs/results_wo_External_{current_time}"

training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="no",
    save_steps=0,
    logging_steps=50,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    save_total_limit=None,
    load_best_model_at_end=False,
    metric_for_best_model=None,
    greater_is_better=None,
    remove_unused_columns=False,
    report_to="none",
)

trainer = ContrastiveNERTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    contrastive_weight=0.1,
)

print(f"Start ablation5 (w/o external POS/BMES). Output: {output_dir}")
trainer.train()
test_results = trainer.evaluate(test_dataset)
print(test_results)

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
    f.write("========== w/o External POS/BMES Test Set Evaluation ==========\n")
    for key, value in test_results.items():
        f.write(f"{key}: {value}\n")

log_history = trainer.state.log_history
train_steps, train_loss, eval_steps, eval_f1 = [], [], [], []
for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    if "eval_f1" in log and "step" in log:
        eval_steps.append(log["step"])
        eval_f1.append(log["eval_f1"])

if train_loss:
    plt.figure(figsize=(8, 6))
    plt.plot(train_steps, train_loss, color="#1f77b4", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_loss.png"), dpi=300)
    plt.close()

if eval_f1:
    plt.figure(figsize=(8, 6))
    plt.plot(eval_steps, eval_f1, color="#ff7f0e", marker="o", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("F1 Score")
    plt.title("Validation F1 (Step-wise)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "validation_f1.png"), dpi=300)
    plt.close()

print("ablation5 finished")






