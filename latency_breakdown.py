import argparse
import datetime
import json
import os
import random
import time
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.utils.data import random_split
from transformers import BertForTokenClassification, BertTokenizerFast
import jieba.posseg as pseg


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

    def forward(self, input_ids, attention_mask, token_type_ids, lexical_ids, structural_ids, labels=None):
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
            output_hidden_states=False,
            return_dict=True,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/formatted_data_fixed.json")
    p.add_argument("--model_path", default="./model_path/chinese-roberta-wwm-ext")
    p.add_argument("--checkpoint", default="./outputs/results_MVCL_GATED_20260312_095653/checkpoint-3100/model.safetensors")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
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
        # fallback to safe placeholders
        bmes_tags = ["S"] * len(text)
        pos_tags = ["[PAD]"] * len(text)

    lexical = [bmes2id.get(t, 0) for t in bmes_tags]
    structural = [pos2id.get(t, 0) for t in pos_tags]
    return lexical, structural


def encode_for_model(tokenizer, words, lexical_seq, structural_seq, max_length):
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    lexical_ids, structural_ids = [], []
    word_ids = encoding.word_ids(batch_index=0)
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            lexical_ids.append(0)
            structural_ids.append(0)
        elif word_idx != previous_word_idx:
            lexical_ids.append(lexical_seq[word_idx] if word_idx < len(lexical_seq) else 0)
            structural_ids.append(structural_seq[word_idx] if word_idx < len(structural_seq) else 0)
        else:
            lexical_ids.append(0)
            structural_ids.append(0)
        previous_word_idx = word_idx

    encoding["lexical_ids"] = torch.tensor([lexical_ids])
    encoding["structural_ids"] = torch.tensor([structural_ids])
    return encoding


def p50(values):
    arr = sorted(values)
    if not arr:
        return 0.0
    n = len(arr)
    return arr[n // 2]


def p95(values):
    arr = sorted(values)
    if not arr:
        return 0.0
    idx = min(len(arr) - 1, int(0.95 * len(arr)))
    return arr[idx]


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if text:
            texts.append(text)

    indices = list(range(len(texts)))
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    test_size = len(indices) - train_size - val_size
    _, _, test_indices = random_split(indices, [train_size, val_size, test_size])
    test_texts = [texts[i] for i in test_indices]

    if args.num_samples > 0:
        test_texts = test_texts[: args.num_samples]

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)

    if args.checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint)
    else:
        state_dict = torch.load(args.checkpoint, map_location="cpu")
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

    preprocess_ms = []
    tokenize_align_ms = []
    forward_ms = []

    all_texts = test_texts
    if not all_texts:
        raise RuntimeError("No samples selected for latency measurement.")

    with torch.no_grad():
        for i, text in enumerate(all_texts):
            words = list(text)

            t0 = time.perf_counter()
            lex_seq, struct_seq = get_lexical_structural_features(text, bmes2id, pos2id)
            # keep ids in embedding bounds
            if num_structural > 0:
                struct_seq = [min(x, num_structural - 1) for x in struct_seq]
            if num_lexical > 0:
                lex_seq = [min(x, num_lexical - 1) for x in lex_seq]
            t1 = time.perf_counter()

            batch = encode_for_model(tokenizer, words, lex_seq, struct_seq, args.max_length)
            t2 = time.perf_counter()

            batch = {k: v.to(device) for k, v in batch.items()}

            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                lexical_ids=batch["lexical_ids"],
                structural_ids=batch["structural_ids"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            # skip warmup samples from stats
            if i >= args.warmup:
                preprocess_ms.append((t1 - t0) * 1000)
                tokenize_align_ms.append((t2 - t1) * 1000)
                forward_ms.append((t4 - t3) * 1000)

    total_without_ext = [a + b for a, b in zip(tokenize_align_ms, forward_ms)]
    total_with_ext = [a + b + c for a, b, c in zip(preprocess_ms, tokenize_align_ms, forward_ms)]

    report = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "num_samples_total": len(all_texts),
        "warmup_skipped": min(args.warmup, len(all_texts)),
        "num_samples_counted": len(total_with_ext),
        "latency_ms": {
            "external_preprocess_mean": mean(preprocess_ms),
            "tokenize_align_mean": mean(tokenize_align_ms),
            "model_forward_mean": mean(forward_ms),
            "total_without_external_preprocess_mean": mean(total_without_ext),
            "total_with_external_preprocess_mean": mean(total_with_ext),
            "total_with_external_preprocess_p50": p50(total_with_ext),
            "total_with_external_preprocess_p95": p95(total_with_ext),
        },
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latency_breakdown_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved latency report to: {out_path}")


if __name__ == "__main__":
    main()

