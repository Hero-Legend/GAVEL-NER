import argparse
import datetime
import json
import os
import random
from statistics import mean, pstdev

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
from transformers import BertForTokenClassification, BertTokenizerFast
from seqeval.metrics import classification_report
import jieba.posseg as pseg
from safetensors.torch import load_file


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MVCL_BERT(nn.Module):
    def __init__(self, model_path, num_labels, num_lexical, num_structural):
        super().__init__()
        self.bert_for_token_cls = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
        hidden_size = self.bert_for_token_cls.config.hidden_size
        self.lexical_embedding = nn.Embedding(num_lexical, hidden_size)
        self.structural_embedding = nn.Embedding(num_structural, hidden_size)
        self.dropout = nn.Dropout(0.3)
        nn.init.zeros_(self.lexical_embedding.weight)
        nn.init.zeros_(self.structural_embedding.weight)
        self.gate_lex = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_struct = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, lexical_ids, structural_ids):
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
            return_dict=True,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/formatted_data_fixed.json")
    p.add_argument("--model_path", default="./model_path/chinese-roberta-wwm-ext")
    p.add_argument("--checkpoint", default="./outputs/results_MVCL_GATED_20260312_095653/checkpoint-3100/model.safetensors")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--noise_levels", default="0.0,0.1,0.2,0.3")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_lexical_structural_features(text, bmes2id, pos2id):
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
        pos_tags = ["[PAD]"] * len(text)

    lexical = [bmes2id.get(t, 0) for t in bmes_tags]
    structural = [pos2id.get(t, 0) for t in pos_tags]
    return lexical, structural


def maybe_corrupt(seq, noise_ratio, vocab_size, rng):
    if noise_ratio <= 0 or vocab_size <= 1:
        return seq
    out = list(seq)
    upper = max(1, vocab_size - 1)
    for i, cur in enumerate(out):
        if rng.random() < noise_ratio:
            if upper == 1:
                out[i] = 1
            else:
                new_val = rng.randint(1, upper)
                if new_val == cur:
                    new_val = (new_val % upper) + 1
                out[i] = new_val
    return out


def build_label_seq(item):
    text = item.get("context") or item.get("text") or ""
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
            except Exception:
                continue
            if start >= len(label_seq) or end > len(label_seq) or start >= end:
                continue
            label_seq[start] = f"B-{entity_type}"
            for i in range(start + 1, end):
                label_seq[i] = f"I-{entity_type}"
    return label_seq


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if not text:
            continue
        labels = build_label_seq(item)
        samples.append((text, labels))

    unique_labels = set(tag for _, ys in samples for tag in ys)
    label2id = {tag: i for i, tag in enumerate(sorted(unique_labels))}
    id2label = {i: tag for tag, i in label2id.items()}

    idx = list(range(len(samples)))
    train_size = int(0.8 * len(idx))
    val_size = int(0.1 * len(idx))
    test_size = len(idx) - train_size - val_size
    split_gen = torch.Generator().manual_seed(args.seed)
    _, _, test_idx = random_split(idx, [train_size, val_size, test_size], generator=split_gen)
    test_samples = [samples[i] for i in test_idx]

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    state_dict = load_file(args.checkpoint)
    num_labels = state_dict["bert_for_token_cls.classifier.weight"].shape[0]
    num_lexical = state_dict["lexical_embedding.weight"].shape[0]
    num_structural = state_dict["structural_embedding.weight"].shape[0]

    model = MVCL_BERT(args.model_path, num_labels, num_lexical, num_structural)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    bmes2id = {"[PAD]": 0, "B": 1, "M": 2, "E": 3, "S": 4}
    pos2id = {"[PAD]": 0}


    noise_levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]

    results = []

    for nl in noise_levels:
        run_metrics = []
        run_times = []
        runs = 1 if nl == 0 else args.repeats

        for run in range(runs):
            rng = random.Random(args.seed + run * 1000 + int(nl * 1000))
            true_labels_all = []
            pred_labels_all = []

            start_t = datetime.datetime.now()

            with torch.no_grad():
                for text, gold_tags in test_samples:
                    lex_seq, struct_seq = get_lexical_structural_features(text, bmes2id, pos2id)

                    lex_seq = [min(x, num_lexical - 1) if num_lexical > 0 else 0 for x in lex_seq]
                    struct_seq = [min(x, num_structural - 1) if num_structural > 0 else 0 for x in struct_seq]

                    lex_seq = maybe_corrupt(lex_seq, nl, num_lexical, rng)
                    struct_seq = maybe_corrupt(struct_seq, nl, num_structural, rng)

                    words = list(text)
                    encoding = tokenizer(
                        words,
                        is_split_into_words=True,
                        truncation=True,
                        padding="max_length",
                        max_length=args.max_length,
                        return_tensors="pt",
                    )

                    label_ids, lexical_ids, structural_ids = [], [], []
                    word_ids = encoding.word_ids(batch_index=0)
                    prev = None
                    for wid in word_ids:
                        if wid is None:
                            label_ids.append(-100)
                            lexical_ids.append(0)
                            structural_ids.append(0)
                        elif wid != prev:
                            label_ids.append(label2id[gold_tags[wid]])
                            lexical_ids.append(lex_seq[wid] if wid < len(lex_seq) else 0)
                            structural_ids.append(struct_seq[wid] if wid < len(struct_seq) else 0)
                        else:
                            label_ids.append(-100)
                            lexical_ids.append(0)
                            structural_ids.append(0)
                        prev = wid

                    batch = {
                        "input_ids": encoding["input_ids"].to(device),
                        "attention_mask": encoding["attention_mask"].to(device),
                        "token_type_ids": encoding["token_type_ids"].to(device),
                        "lexical_ids": torch.tensor([lexical_ids], device=device),
                        "structural_ids": torch.tensor([structural_ids], device=device),
                    }

                    outputs = model(**batch)
                    pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                    label_ids_np = np.array(label_ids)

                    true_seq = [id2label[l] for l in label_ids_np if l != -100]
                    pred_seq = [id2label[p] for p, l in zip(pred, label_ids_np) if l != -100]
                    true_labels_all.append(true_seq)
                    pred_labels_all.append(pred_seq)

            end_t = datetime.datetime.now()
            sec = (end_t - start_t).total_seconds()
            run_times.append(sec)

            rep = classification_report(true_labels_all, pred_labels_all, output_dict=True)
            run_metrics.append(
                {
                    "precision": rep["macro avg"]["precision"],
                    "recall": rep["macro avg"]["recall"],
                    "f1": rep["macro avg"]["f1-score"],
                }
            )

        p_vals = [m["precision"] for m in run_metrics]
        r_vals = [m["recall"] for m in run_metrics]
        f_vals = [m["f1"] for m in run_metrics]

        results.append(
            {
                "noise_ratio": nl,
                "runs": runs,
                "precision_mean": mean(p_vals),
                "recall_mean": mean(r_vals),
                "f1_mean": mean(f_vals),
                "f1_std": 0.0 if len(f_vals) == 1 else pstdev(f_vals),
                "runtime_sec_mean": mean(run_times),
            }
        )

    clean_f1 = next(x["f1_mean"] for x in results if abs(x["noise_ratio"]) < 1e-8)
    for row in results:
        row["f1_drop_abs"] = clean_f1 - row["f1_mean"]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"noise_robustness_{ts}.json")
    out_md = os.path.join(out_dir, f"noise_robustness_{ts}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "device": str(device),
                "test_size": len(test_samples),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lines = [
        "| Noise Ratio | F1 (mean) | F1 std | F1 drop vs clean | Precision | Recall |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['noise_ratio']:.2f} | {row['f1_mean']*100:.2f} | {row['f1_std']*100:.2f} | {row['f1_drop_abs']*100:.2f} | {row['precision_mean']*100:.2f} | {row['recall_mean']*100:.2f} |"
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()

