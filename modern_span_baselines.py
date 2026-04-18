import copy
import datetime
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertModel, BertTokenizerFast


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


MODEL_PATH = os.environ.get("CAIL_MODEL_PATH", "./model_path/chinese-roberta-wwm-ext")
DATA_PATH = os.environ.get("CAIL_DATA_PATH", "./data/formatted_data_fixed.json")
RUN_TAG = os.environ.get("SPAN_RUN_TAG", "modern_span_baselines")
MAX_LENGTH = int(os.environ.get("SPAN_MAX_LENGTH", "256"))
EPOCHS = int(float(os.environ.get("SPAN_EPOCHS", "12")))
TRAIN_BATCH = int(os.environ.get("SPAN_BATCH", "4"))
EVAL_BATCH = int(os.environ.get("SPAN_EVAL_BATCH", "8"))
GRAD_ACC = int(os.environ.get("SPAN_GRAD_ACC", "4"))
BERT_LR = float(os.environ.get("SPAN_BERT_LR", "2e-5"))
HEAD_LR = float(os.environ.get("SPAN_HEAD_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("SPAN_WEIGHT_DECAY", "0.01"))
SEED = int(os.environ.get("SPAN_SEED", "42"))
SMOKE_LIMIT = int(os.environ.get("SPAN_SMOKE_LIMIT", "0"))
METHODS = [x.strip() for x in os.environ.get("SPAN_METHODS", "global_pointer,biaffine_span").split(",") if x.strip()]
SPLIT_MODE = os.environ.get("SPAN_SPLIT_MODE", "python_shuffle")


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


def load_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    entity_types = set()
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if not text:
            continue

        label_seq = ["O"] * len(text)
        spans = []
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

                entity_types.add(entity_type)
                spans.append((start, end - 1, entity_type))
                label_seq[start] = f"B-{entity_type}"
                for i in range(start + 1, end):
                    label_seq[i] = f"I-{entity_type}"

        samples.append({"chars": list(text), "labels": label_seq, "spans": spans})

    return samples, sorted(entity_types)


class SpanNERDataset(Dataset):
    def __init__(self, samples, tokenizer, entity2id: Dict[str, int], max_length: int):
        self.samples = samples
        self.tokenizer = tokenizer
        self.entity2id = entity2id
        self.max_length = max_length
        self.num_entities = len(entity2id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        chars = sample["chars"]
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        char_to_token = {}
        valid_token_mask = torch.zeros(self.max_length, dtype=torch.bool)
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                char_to_token[word_idx] = token_idx
                valid_token_mask[token_idx] = True
            previous_word_idx = word_idx

        span_labels = torch.zeros(self.num_entities, self.max_length, self.max_length, dtype=torch.float)
        for start, end, ent_type in sample["spans"]:
            if start in char_to_token and end in char_to_token:
                s_tok = char_to_token[start]
                e_tok = char_to_token[end]
                if s_tok <= e_tok:
                    span_labels[self.entity2id[ent_type], s_tok, e_tok] = 1.0

        out = {k: v.squeeze(0) for k, v in encoding.items()}
        out["span_labels"] = span_labels
        out["valid_token_mask"] = valid_token_mask
        out["idx"] = torch.tensor(idx)
        return out


def apply_rope(x):
    # x: [batch, seq, heads, dim]
    dim = x.size(-1)
    pos = torch.arange(x.size(1), dtype=torch.float, device=x.device)
    idx = torch.arange(0, dim, 2, dtype=torch.float, device=x.device)
    inv_freq = torch.pow(10000.0, -idx / dim)
    sinusoid = torch.einsum("n,d->nd", pos, inv_freq)
    sin = sinusoid.sin()[None, :, None, :]
    cos = sinusoid.cos()[None, :, None, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class GlobalPointerNER(nn.Module):
    def __init__(self, model_path: str, num_entities: int, inner_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.num_entities = num_entities
        self.inner_dim = inner_dim
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, num_entities * inner_dim * 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = self.dropout(outputs.last_hidden_state)
        projected = self.dense(hidden)
        projected = projected.view(hidden.size(0), hidden.size(1), self.num_entities, self.inner_dim * 2)
        qw, kw = projected[..., : self.inner_dim], projected[..., self.inner_dim :]
        qw = apply_rope(qw)
        kw = apply_rope(kw)
        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw) / math.sqrt(self.inner_dim)
        return logits


class BiaffineSpanNER(nn.Module):
    def __init__(self, model_path: str, num_entities: int, biaffine_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.start_mlp = nn.Sequential(nn.Linear(hidden, biaffine_dim), nn.GELU(), nn.Dropout(dropout))
        self.end_mlp = nn.Sequential(nn.Linear(hidden, biaffine_dim), nn.GELU(), nn.Dropout(dropout))
        self.U = nn.Parameter(torch.empty(num_entities, biaffine_dim + 1, biaffine_dim + 1))
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = self.dropout(outputs.last_hidden_state)
        start = self.start_mlp(hidden)
        end = self.end_mlp(hidden)
        ones = torch.ones(*start.shape[:-1], 1, device=start.device, dtype=start.dtype)
        start = torch.cat([start, ones], dim=-1)
        end = torch.cat([end, ones], dim=-1)
        return torch.einsum("bxi,eij,byj->bexy", start, self.U, end)


def mask_logits(logits, attention_mask, valid_token_mask):
    seq_len = logits.size(-1)
    token_mask = attention_mask.bool() & valid_token_mask.bool()
    pair_mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
    upper = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=logits.device), diagonal=0)
    pair_mask = pair_mask & upper[None, None, :, :]
    return logits.masked_fill(~pair_mask, -1e12), pair_mask


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def span_loss(logits, labels, attention_mask, valid_token_mask):
    logits, _ = mask_logits(logits, attention_mask, valid_token_mask)
    return multilabel_categorical_crossentropy(logits.flatten(1), labels.flatten(1))


def spans_to_bio(spans: List[Tuple[int, int, str, float]], n_chars: int) -> List[str]:
    labels = ["O"] * n_chars
    occupied = [False] * n_chars
    for start, end, ent_type, score in sorted(spans, key=lambda x: (x[3], x[1] - x[0]), reverse=True):
        if start < 0 or end >= n_chars or start > end:
            continue
        if any(occupied[i] for i in range(start, end + 1)):
            continue
        labels[start] = f"B-{ent_type}"
        occupied[start] = True
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{ent_type}"
            occupied[i] = True
    return labels


@torch.no_grad()
def predict_dataset(model, dataset, loader, id2entity, threshold: float, device):
    model.eval()
    predictions = [None] * len(dataset)
    for batch in loader:
        idxs = batch.pop("idx").tolist()
        span_labels = batch.pop("span_labels")
        valid_token_mask = batch.pop("valid_token_mask").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)
        logits, _ = mask_logits(logits, batch["attention_mask"], valid_token_mask)
        logits = logits.detach().cpu()

        for b, local_idx in enumerate(idxs):
            sample = dataset.samples[local_idx]
            chars = sample["chars"]
            encoding = dataset.tokenizer(
                chars,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=dataset.max_length,
                return_tensors="pt",
            )
            token_to_char = {}
            prev = None
            for tok_idx, word_idx in enumerate(encoding.word_ids(batch_index=0)):
                if word_idx is not None and word_idx != prev:
                    token_to_char[tok_idx] = word_idx
                prev = word_idx

            pred_spans = []
            ent_ids, starts, ends = torch.where(logits[b] > threshold)
            for ent_id, s_tok, e_tok in zip(ent_ids.tolist(), starts.tolist(), ends.tolist()):
                if s_tok in token_to_char and e_tok in token_to_char:
                    pred_spans.append((token_to_char[s_tok], token_to_char[e_tok], id2entity[ent_id], float(logits[b, ent_id, s_tok, e_tok])))
            predictions[local_idx] = spans_to_bio(pred_spans, len(chars))

    return predictions


def evaluate_model(model, dataset, subset_indices, id2entity, threshold: float, device, batch_size: int):
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    preds_subset = predict_dataset(model, dataset, loader, id2entity, threshold, device)
    true_labels, pred_labels = [], []
    for idx in subset_indices:
        true_labels.append(dataset.samples[idx]["labels"])
        pred_labels.append(preds_subset[idx])
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
        "threshold": threshold,
    }


def tune_threshold(model, dataset, val_idx, id2entity, device):
    candidates = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
    scored = [evaluate_model(model, dataset, val_idx, id2entity, t, device, EVAL_BATCH) for t in candidates]
    return max(scored, key=lambda x: x["f1"]), scored


@dataclass
class RunResult:
    method: str
    best_epoch: int
    best_val_f1: float
    threshold: float
    test_precision: float
    test_recall: float
    test_f1: float


def make_model(method: str, model_path: str, num_entities: int):
    if method == "global_pointer":
        return GlobalPointerNER(model_path, num_entities)
    if method == "biaffine_span":
        return BiaffineSpanNER(model_path, num_entities)
    raise ValueError(f"Unknown method: {method}")


def make_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight"]
    bert_params = [(n, p) for n, p in model.named_parameters() if n.startswith("bert.")]
    other_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("bert.")]
    groups = [
        {"params": [p for n, p in bert_params if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY, "lr": BERT_LR},
        {"params": [p for n, p in bert_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": BERT_LR},
        {"params": [p for n, p in other_params if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY, "lr": HEAD_LR},
        {"params": [p for n, p in other_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": HEAD_LR},
    ]
    return torch.optim.AdamW(groups)


def train_one(method, dataset, splits, id2entity, output_dir, device):
    train_idx, val_idx, test_idx = splits
    model = make_model(method, MODEL_PATH, len(id2entity)).to(device)
    optimizer = make_optimizer(model)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=TRAIN_BATCH, shuffle=True)

    best_state = None
    best_epoch = -1
    best_val = {"f1": -1.0, "threshold": 0.0}
    logs = []
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("idx")
            labels = batch.pop("span_labels").to(device)
            valid_token_mask = batch.pop("valid_token_mask").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch)
            loss = span_loss(logits, labels, batch["attention_mask"], valid_token_mask)
            (loss / GRAD_ACC).backward()
            running_loss += float(loss.detach().cpu())

            if step % GRAD_ACC == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        val_best, threshold_scores = tune_threshold(model, dataset, val_idx, id2entity, device)
        avg_loss = running_loss / max(1, len(train_loader))
        log_row = {"epoch": epoch, "train_loss": avg_loss, "val": val_best, "threshold_scores": threshold_scores}
        logs.append(log_row)
        print(f"[{method}] epoch={epoch} loss={avg_loss:.4f} val_f1={val_best['f1']:.4f} threshold={val_best['threshold']}", flush=True)

        if val_best["f1"] > best_val["f1"]:
            best_val = val_best
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate_model(model, dataset, test_idx, id2entity, best_val["threshold"], device, EVAL_BATCH)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(best_state, os.path.join(output_dir, "best_state.pt"))
    with open(os.path.join(output_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "evaluation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"method: {method}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"best_val_f1: {best_val['f1']}\n")
        for k, v in test_metrics.items():
            f.write(f"test_{k}: {v}\n")

    del model
    torch.cuda.empty_cache()
    return RunResult(
        method=method,
        best_epoch=best_epoch,
        best_val_f1=best_val["f1"],
        threshold=best_val["threshold"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_f1=test_metrics["f1"],
    )


def build_splits(n_items: int, seed: int):
    train_size = int(0.8 * n_items)
    val_size = int(0.1 * n_items)
    test_size = n_items - train_size - val_size
    if SPLIT_MODE == "torch_random_split":
        generator = torch.Generator().manual_seed(seed)
        split = torch.utils.data.random_split(list(range(n_items)), [train_size, val_size, test_size], generator=generator)
        return [int(x) for x in split[0]], [int(x) for x in split[1]], [int(x) for x in split[2]]
    indices = list(range(n_items))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:train_size], indices[train_size : train_size + val_size], indices[train_size + val_size :]


def main():
    set_seed(SEED)
    samples, entity_types = load_data(DATA_PATH)
    if SMOKE_LIMIT > 0:
        samples = samples[:SMOKE_LIMIT]
        print(f"Smoke mode: using {len(samples)} samples", flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    entity2id = {x: i for i, x in enumerate(entity_types)}
    id2entity = {i: x for x, i in entity2id.items()}
    dataset = SpanNERDataset(samples, tokenizer, entity2id, MAX_LENGTH)
    splits = build_splits(len(dataset), SEED)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = f"./outputs/{RUN_TAG}_{timestamp}"
    os.makedirs(output_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} samples={len(dataset)} train/val/test={[len(x) for x in splits]}", flush=True)
    print(f"methods={METHODS}", flush=True)

    results = []
    for method in METHODS:
        method_dir = os.path.join(output_root, method)
        result = train_one(method, dataset, splits, id2entity, method_dir, device)
        results.append(result.__dict__)

    summary = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "seed": SEED,
        "split_mode": SPLIT_MODE,
        "max_length": MAX_LENGTH,
        "epochs": EPOCHS,
        "train_batch": TRAIN_BATCH,
        "eval_batch": EVAL_BATCH,
        "grad_acc": GRAD_ACC,
        "entity_types": entity_types,
        "split_sizes": {"train": len(splits[0]), "validation": len(splits[1]), "test": len(splits[2])},
        "results": results,
    }

    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_root, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Contemporary Span-based Baselines on CAIL2021\n\n")
        f.write(f"- Backbone: `{MODEL_PATH}`\n")
        f.write(f"- Data: `{DATA_PATH}`\n")
        f.write(f"- Seed: `{SEED}`\n")
        f.write(f"- Split mode: `{SPLIT_MODE}`\n")
        f.write(f"- Split sizes: train={len(splits[0])}, validation={len(splits[1])}, test={len(splits[2])}\n\n")
        f.write("| Method | P | R | F1 | Best Epoch | Threshold |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for row in results:
            f.write(
                f"| {row['method']} | {row['test_precision']:.4f} | {row['test_recall']:.4f} | {row['test_f1']:.4f} | {row['best_epoch']} | {row['threshold']:.2f} |\n"
            )

    print(f"Done. Summary saved to {output_root}/summary.md", flush=True)


if __name__ == "__main__":
    main()
