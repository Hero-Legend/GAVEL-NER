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
from transformers import BertModel, BertTokenizerFast
import jieba.posseg as pseg
from safetensors.torch import load_file
from torchcrf import CRF


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MVCL_BERT_CRF(nn.Module):
    def __init__(self, model_path, num_labels, num_lexical, num_structural):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        hidden_size = self.bert.config.hidden_size

        self.lexical_embedding = nn.Embedding(num_lexical, hidden_size)
        self.structural_embedding = nn.Embedding(num_structural, hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.gate_lex = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_struct = nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_tags=num_labels, batch_first=True)

    def forward_infer(self, input_ids, attention_mask, token_type_ids, lexical_ids, structural_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]

        lex_embeds = self.lexical_embedding(lexical_ids)
        struct_embeds = self.structural_embedding(structural_ids)

        g_lex = torch.sigmoid(self.gate_lex(torch.cat([sequence_output, lex_embeds], dim=-1)))
        fused_1 = sequence_output + g_lex * lex_embeds

        g_struct = torch.sigmoid(self.gate_struct(torch.cat([fused_1, struct_embeds], dim=-1)))
        fused_embeds = fused_1 + g_struct * struct_embeds

        fused_embeds = self.dropout(fused_embeds)
        logits = self.classifier(fused_embeds)
        mask = attention_mask.bool()
        _ = self.crf.decode(logits, mask=mask)
        return logits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="./data/formatted_data_fixed.json")
    p.add_argument("--model_path", default="./model_path/chinese-roberta-wwm-ext")
    p.add_argument("--checkpoint", default="./outputs/results_MVCL_CRF_FIXED_20260314_093240/checkpoint-3100/model.safetensors")
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
    prev = None
    for wid in word_ids:
        if wid is None:
            lexical_ids.append(0)
            structural_ids.append(0)
        elif wid != prev:
            lexical_ids.append(lexical_seq[wid] if wid < len(lexical_seq) else 0)
            structural_ids.append(structural_seq[wid] if wid < len(structural_seq) else 0)
        else:
            lexical_ids.append(0)
            structural_ids.append(0)
        prev = wid

    encoding["lexical_ids"] = torch.tensor([lexical_ids])
    encoding["structural_ids"] = torch.tensor([structural_ids])
    return encoding


def p50(values):
    arr = sorted(values)
    return arr[len(arr) // 2] if arr else 0.0


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
    texts = [x.get("context") or x.get("text") or "" for x in data]
    texts = [t for t in texts if t]

    idx = list(range(len(texts)))
    train_size = int(0.8 * len(idx))
    val_size = int(0.1 * len(idx))
    test_size = len(idx) - train_size - val_size
    gen = torch.Generator().manual_seed(args.seed)
    _, _, test_idx = torch.utils.data.random_split(idx, [train_size, val_size, test_size], generator=gen)
    test_texts = [texts[i] for i in test_idx]
    if args.num_samples > 0:
        test_texts = test_texts[: args.num_samples]

    tokenizer = BertTokenizerFast.from_pretrained(args.model_path)
    state_dict = load_file(args.checkpoint)
    num_labels = state_dict["classifier.weight"].shape[0]
    num_lexical = state_dict["lexical_embedding.weight"].shape[0]
    num_structural = state_dict["structural_embedding.weight"].shape[0]

    model = MVCL_BERT_CRF(args.model_path, num_labels, num_lexical, num_structural)
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    bmes2id = {"[PAD]": 0, "B": 1, "M": 2, "E": 3, "S": 4}
    pos2id = {"[PAD]": 0}

    preprocess_ms, tokenize_ms, forward_ms = [], [], []

    with torch.no_grad():
        for i, text in enumerate(test_texts):
            words = list(text)

            t0 = time.perf_counter()
            lex_seq, struct_seq = get_lexical_structural_features(text, bmes2id, pos2id)
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
            _ = model.forward_infer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                lexical_ids=batch["lexical_ids"],
                structural_ids=batch["structural_ids"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()

            if i >= args.warmup:
                preprocess_ms.append((t1 - t0) * 1000)
                tokenize_ms.append((t2 - t1) * 1000)
                forward_ms.append((t4 - t3) * 1000)

    total_wo = [a + b for a, b in zip(tokenize_ms, forward_ms)]
    total_w = [a + b + c for a, b, c in zip(preprocess_ms, tokenize_ms, forward_ms)]

    report = {
        "device": str(device),
        "checkpoint": args.checkpoint,
        "num_samples_total": len(test_texts),
        "warmup_skipped": min(args.warmup, len(test_texts)),
        "num_samples_counted": len(total_w),
        "latency_ms": {
            "external_preprocess_mean": mean(preprocess_ms),
            "tokenize_align_mean": mean(tokenize_ms),
            "model_forward_with_crf_decode_mean": mean(forward_ms),
            "total_without_external_preprocess_mean": mean(total_wo),
            "total_with_external_preprocess_mean": mean(total_w),
            "total_with_external_preprocess_p50": p50(total_w),
            "total_with_external_preprocess_p95": p95(total_w),
        },
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("./outputs", f"latency_breakdown_crf_{ts}.json")
    os.makedirs("./outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
