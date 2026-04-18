import datetime
import importlib
import json
import os
import random
import sys
from statistics import mean, pstdev

import jieba.posseg as pseg
import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import random_split
from transformers import BertForTokenClassification, BertTokenizerFast
from seqeval.metrics import classification_report


VENDOR_NLP = os.path.join(os.path.dirname(__file__), "vendor_nlp")
if os.path.isdir(VENDOR_NLP) and VENDOR_NLP not in sys.path:
    sys.path.insert(0, VENDOR_NLP)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GatedMVCLBert(nn.Module):
    def __init__(self, model_path, num_labels, num_lexical, num_structural):
        super().__init__()
        self.bert_for_token_cls = BertForTokenClassification.from_pretrained(model_path, num_labels=num_labels)
        hidden_size = self.bert_for_token_cls.config.hidden_size
        self.lexical_embedding = nn.Embedding(num_lexical, hidden_size)
        self.structural_embedding = nn.Embedding(num_structural, hidden_size)
        self.gate_lex = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_struct = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids, lexical_ids, structural_ids, return_gates=False):
        inputs_embeds = self.bert_for_token_cls.bert.embeddings.word_embeddings(input_ids)
        lex_embeds = self.lexical_embedding(lexical_ids)
        struct_embeds = self.structural_embedding(structural_ids)
        g_lex = torch.sigmoid(self.gate_lex(torch.cat([inputs_embeds, lex_embeds], dim=-1)))
        fused_1 = inputs_embeds + g_lex * lex_embeds
        g_struct = torch.sigmoid(self.gate_struct(torch.cat([fused_1, struct_embeds], dim=-1)))
        fused_embeds = fused_1 + g_struct * struct_embeds
        fused_embeds = self.dropout(fused_embeds)

        outputs = self.bert_for_token_cls(
            inputs_embeds=fused_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        if return_gates:
            outputs.g_lex = g_lex
            outputs.g_struct = g_struct
        return outputs


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


def maybe_corrupt(seq, noise_ratio, vocab_size, rng):
    if noise_ratio <= 0 or vocab_size <= 1:
        return list(seq)
    out = list(seq)
    upper = max(1, vocab_size - 1)
    for i, cur in enumerate(out):
        if rng.random() < noise_ratio:
            new_val = rng.randint(1, upper)
            if new_val == cur and upper > 1:
                new_val = (new_val % upper) + 1
            out[i] = new_val
    return out


def get_segmenter(mode):
    if mode == "jieba":
        return lambda text: list(pseg.cut(text))
    if mode == "char":
        return lambda text: [(ch, "x") for ch in text]
    if mode == "thulac":
        module = importlib.import_module("thulac")
        thu = module.thulac(seg_only=False)
        return lambda text: thu.cut(text)
    if mode == "pkuseg":
        module = importlib.import_module("pkuseg")
        seg = module.pkuseg(postag=True)
        return lambda text: seg.cut(text)
    raise ValueError(f"Unsupported mode: {mode}")


def unpack_segment_pair(pair):
    if isinstance(pair, (list, tuple)):
        if len(pair) >= 2:
            return str(pair[0]), str(pair[1])
        if len(pair) == 1:
            return str(pair[0]), "x"

    if hasattr(pair, "word"):
        flag = getattr(pair, "tag", getattr(pair, "flag", "x"))
        return str(pair.word), str(flag)

    try:
        return str(pair[0]), str(pair[1])
    except Exception:
        return str(pair), "x"


def build_features(text, mode, bmes2id, pos2id):
    cutter = get_segmenter(mode)
    bmes_tags, pos_tags = [], []
    for pair in cutter(text):
        word, flag = unpack_segment_pair(pair)
        length = len(word)
        if length == 1:
            bmes_tags.append("S")
        else:
            bmes_tags.extend(["B"] + ["M"] * (length - 2) + ["E"])
        if flag not in pos2id:
            pos2id[flag] = len(pos2id)
        pos_tags.extend([flag] * length)

    if len(bmes_tags) != len(text):
        if "x" not in pos2id:
            pos2id["x"] = len(pos2id)
        bmes_tags = ["S"] * len(text)
        pos_tags = ["x"] * len(text)

    lexical = [bmes2id.get(t, 0) for t in bmes_tags]
    structural = [pos2id.get(t, 0) for t in pos_tags]
    return lexical, structural


def main():
    set_seed(42)
    data_path = "./data/formatted_data_fixed.json"
    model_path = "./model_path/chinese-roberta-wwm-ext"
    checkpoint = "./outputs/results_MVCL_GATED_20260312_095653/checkpoint-3100/model.safetensors"
    modes = ["jieba", "char", "thulac", "pkuseg"]
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    repeats = 3
    max_length = 256

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if text:
            samples.append((text, build_label_seq(item)))

    unique_labels = sorted(set(tag for _, ys in samples for tag in ys))
    label2id = {tag: i for i, tag in enumerate(unique_labels)}
    id2label = {i: tag for tag, i in label2id.items()}

    split_gen = torch.Generator().manual_seed(42)
    idx = list(range(len(samples)))
    train_size = int(0.8 * len(idx))
    val_size = int(0.1 * len(idx))
    test_size = len(idx) - train_size - val_size
    _, _, test_idx = random_split(idx, [train_size, val_size, test_size], generator=split_gen)
    test_samples = [samples[i] for i in test_idx.indices]

    state_dict = load_file(checkpoint)
    num_labels = state_dict["bert_for_token_cls.classifier.weight"].shape[0]
    num_lexical = state_dict["lexical_embedding.weight"].shape[0]
    num_structural = state_dict["structural_embedding.weight"].shape[0]

    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model = GatedMVCLBert(model_path, num_labels, num_lexical, num_structural)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    bmes2id = {"[PAD]": 0, "B": 1, "M": 2, "E": 3, "S": 4}
    pos2id = {"[PAD]": 0}
    results = []

    for mode in modes:
        try:
            get_segmenter(mode)
        except Exception as exc:
            results.append({"mode": mode, "status": "missing", "error": repr(exc)})
            continue

        for noise_ratio in noise_levels:
            runs = 1 if noise_ratio == 0 else repeats
            metric_runs, lex_gate_runs, struct_gate_runs = [], [], []

            for run in range(runs):
                rng = random.Random(42 + run * 1000 + int(noise_ratio * 1000))
                true_labels_all = []
                pred_labels_all = []
                lex_gate_vals = []
                struct_gate_vals = []

                with torch.no_grad():
                    for text, gold_tags in test_samples:
                        lex_seq, struct_seq = build_features(text, mode, bmes2id, pos2id)
                        lex_seq = [min(x, num_lexical - 1) for x in lex_seq]
                        struct_seq = [min(x, num_structural - 1) for x in struct_seq]
                        lex_seq = maybe_corrupt(lex_seq, noise_ratio, num_lexical, rng)
                        struct_seq = maybe_corrupt(struct_seq, noise_ratio, num_structural, rng)

                        words = list(text)
                        encoding = tokenizer(
                            words,
                            is_split_into_words=True,
                            truncation=True,
                            padding="max_length",
                            max_length=max_length,
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
                        outputs = model(**batch, return_gates=True)
                        pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy()[0]
                        label_ids_np = np.array(label_ids)
                        valid_mask = label_ids_np != -100

                        true_seq = [id2label[l] for l in label_ids_np[valid_mask]]
                        pred_seq = [id2label[p] for p, keep in zip(pred, valid_mask) if keep]
                        true_labels_all.append(true_seq)
                        pred_labels_all.append(pred_seq)
                        lex_gate_vals.append(outputs.g_lex[0][valid_mask].mean().item())
                        struct_gate_vals.append(outputs.g_struct[0][valid_mask].mean().item())

                rep = classification_report(true_labels_all, pred_labels_all, output_dict=True)
                metric_runs.append(rep["macro avg"]["f1-score"])
                lex_gate_runs.append(mean(lex_gate_vals))
                struct_gate_runs.append(mean(struct_gate_vals))

            results.append(
                {
                    "mode": mode,
                    "status": "ok",
                    "noise_ratio": noise_ratio,
                    "f1_mean": mean(metric_runs),
                    "f1_std": pstdev(metric_runs) if len(metric_runs) > 1 else 0.0,
                    "gate_lex_mean": mean(lex_gate_runs),
                    "gate_struct_mean": mean(struct_gate_runs),
                }
            )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/gate_tool_sensitivity_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("# Gate and Tool Sensitivity Analysis\n\n")
        f.write("| Mode | Noise | F1 mean | F1 std | Gate lex mean | Gate struct mean | Status |\n")
        f.write("|---|---:|---:|---:|---:|---:|---|\n")
        for row in results:
            if row["status"] != "ok":
                f.write(f"| {row['mode']} | - | - | - | - | - | missing |\n")
            else:
                f.write(
                    f"| {row['mode']} | {row['noise_ratio']:.2f} | {row['f1_mean']:.4f} | {row['f1_std']:.4f} | {row['gate_lex_mean']:.4f} | {row['gate_struct_mean']:.4f} | ok |\n"
                )


if __name__ == "__main__":
    main()
