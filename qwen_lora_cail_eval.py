import datetime
import json
import os
import random
import re

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


LABEL_CODE_TO_CN = {
    "NHVI": "受害人",
    "NHCS": "嫌疑人或被告人",
    "NCSP": "公检法机关",
    "NCSM": "鉴定或医疗机构",
    "NCGV": "政府行政机关",
    "NT": "时间",
    "NS": "地点",
    "NASI": "赃款赃物",
    "NATS": "作案工具",
    "NO": "其他涉案物品",
}
LABEL_CN_TO_CODE = {v: k for k, v in LABEL_CODE_TO_CN.items()}

SYSTEM_PROMPT = (
    "你是一个专业的中国司法命名实体识别助手。"
    "请从给定司法文本中抽取命名实体，并且只输出 JSON 数组。"
)

CATEGORY_DESC = """【实体类别】
1. 受害人: 受到犯罪侵害的人。
2. 嫌疑人或被告人: 实施犯罪的人。
3. 公检法机关: 公安局、检察院、法院等。
4. 鉴定或医疗机构: 医院、鉴定中心、价格认证中心等。
5. 政府行政机关: 除公检法外的其他政府部门。
6. 时间: 与案件有关的时间表达。
7. 地点: 与案件有关的地点表达。
8. 赃款赃物: 被盗、被抢或涉案的财物。
9. 作案工具: 实施犯罪所用工具。
10. 其他涉案物品: 与案件相关但不属于前述类别的重要物品。"""

OUTPUT_RULES = """【输出要求】
- 只输出 JSON 数组，不要输出解释、分析或 Markdown。
- 每个元素格式为 {"entity": "实体原文", "type": "中文类别名"}。
- `entity` 必须逐字摘自原文，不能改写、不能补字、不能缩写。
- `type` 必须严格使用以上 10 个中文类别名之一。
- 如果没有实体，输出 []。
- 去重后输出。"""


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def normalize_gold_entities(item, text: str):
    entities = []
    seen = set()
    for ent in item.get("entities", []):
        label = ent.get("label") or ent.get("type") or "UNKNOWN"
        raw_spans = ent.get("span", [])
        if len(raw_spans) > 0 and not isinstance(raw_spans[0], list):
            raw_spans = [raw_spans]
        for span in raw_spans:
            try:
                start, end = parse_span(span)
            except Exception:
                continue
            if start >= len(text) or end > len(text) or start >= end:
                continue
            entity_text = text[start:end]
            pair = (entity_text, label)
            if pair not in seen:
                seen.add(pair)
                entities.append((start, entity_text, label))
    entities.sort(key=lambda x: x[0])
    return entities


def build_response_text(item, text: str) -> str:
    rows = []
    for _, entity_text, label in normalize_gold_entities(item, text):
        if label in LABEL_CODE_TO_CN:
            rows.append({"entity": entity_text, "type": LABEL_CODE_TO_CN[label]})
    return json.dumps(rows, ensure_ascii=False)


def build_user_prompt(text: str) -> str:
    return f"""{CATEGORY_DESC}

{OUTPUT_RULES}

【待抽取文本】
{text}
"""


def build_chat_messages(text: str, assistant_text: str | None = None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(text)},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def render_chat_text(tokenizer, messages, add_generation_prompt: bool):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for msg in messages:
        rendered.append(f"<|{msg['role']}|>\n{msg['content']}")
    if add_generation_prompt:
        rendered.append("<|assistant|>\n")
    return "\n".join(rendered)


def load_split(data_path: str, seed: int = 42):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if text:
            valid.append(item)

    set_seed(seed)
    indices = list(range(len(valid)))
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    test_size = len(indices) - train_size - val_size
    train_idx, val_idx, test_idx = random_split(indices, [train_size, val_size, test_size])

    return {
        "train": [valid[i] for i in train_idx.indices],
        "val": [valid[i] for i in val_idx.indices],
        "test": [valid[i] for i in test_idx.indices],
    }


def maybe_limit(items, limit):
    return items if limit <= 0 else items[:limit]


def to_chat_text(tokenizer, item):
    text = item.get("context") or item.get("text") or ""
    assistant_text = build_response_text(item, text)
    return render_chat_text(tokenizer, build_chat_messages(text, assistant_text), add_generation_prompt=False)


def to_prompt_text(tokenizer, item):
    text = item.get("context") or item.get("text") or ""
    return render_chat_text(tokenizer, build_chat_messages(text), add_generation_prompt=True)


def prepare_sft_dataset(tokenizer, items):
    rows = [{"text": to_chat_text(tokenizer, item), "prompt_text": to_prompt_text(tokenizer, item)} for item in items]
    return Dataset.from_list(rows)


def tokenize_function(tokenizer, max_length):
    def _tok(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_length)
        prompt_only = tokenizer(batch["prompt_text"], truncation=True, max_length=max_length)
        labels = []
        for input_ids, prompt_ids in zip(out["input_ids"], prompt_only["input_ids"]):
            masked = input_ids[:]
            prompt_len = min(len(prompt_ids), len(masked))
            masked[:prompt_len] = [-100] * prompt_len
            labels.append(masked)
        out["labels"] = labels
        return out

    return _tok


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return match.group(0) if match else "[]"


def parse_prediction(text: str, original_text: str):
    arr_text = extract_json_array(text)
    try:
        data = json.loads(arr_text)
    except Exception:
        return set(), arr_text
    preds = set()
    if not isinstance(data, list):
        return set(), arr_text
    for item in data:
        if not isinstance(item, dict):
            continue
        ent = str(item.get("entity", "")).strip()
        typ_cn = str(item.get("type", "")).strip()
        if not ent or typ_cn not in LABEL_CN_TO_CODE:
            continue
        if ent not in original_text:
            continue
        preds.add((ent, LABEL_CN_TO_CODE[typ_cn]))
    return preds, arr_text


def build_generation_prompt(tokenizer, text: str):
    return render_chat_text(tokenizer, build_chat_messages(text), add_generation_prompt=True)


def evaluate_generation(model, tokenizer, test_items, output_dir):
    device = next(model.parameters()).device
    max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
    true_positives = 0
    pred_positives = 0
    actual_positives = 0
    samples = []

    model.eval()
    with torch.no_grad():
        for idx, item in enumerate(test_items):
            text = item.get("context") or item.get("text") or ""
            prompt = build_generation_prompt(tokenizer, text)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            pred_set, raw_json = parse_prediction(completion, text)

            gold_set = set((entity_text, label) for _, entity_text, label in normalize_gold_entities(item, text))
            true_positives += len(pred_set.intersection(gold_set))
            pred_positives += len(pred_set)
            actual_positives += len(gold_set)

            if idx < 10:
                samples.append(
                    {
                        "text": text,
                        "gold": sorted(list(gold_set)),
                        "pred": sorted(list(pred_set)),
                        "raw_output": completion,
                        "parsed_json": raw_json,
                    }
                )

    precision = true_positives / pred_positives if pred_positives > 0 else 0.0
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "pred_positives": pred_positives,
        "actual_positives": actual_positives,
        "test_size": len(test_items),
    }
    with open(os.path.join(output_dir, "strict_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "sample_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "strict_metrics.md"), "w", encoding="utf-8") as f:
        f.write("# Qwen LoRA Strict Evaluation\n\n")
        f.write(f"- Precision: {precision:.4f}\n")
        f.write(f"- Recall: {recall:.4f}\n")
        f.write(f"- F1: {f1:.4f}\n")
        f.write(f"- TP / Pred / Gold: {true_positives} / {pred_positives} / {actual_positives}\n")
        f.write(f"- Test size: {len(test_items)}\n")
    return metrics


def main():
    set_seed(42)
    default_local_model = "/gz-fs/storage/models/Qwen2.5-7B-Instruct"
    model_name = os.environ.get(
        "QWEN_MODEL",
        default_local_model if os.path.isdir(default_local_model) else "Qwen/Qwen2.5-1.5B-Instruct",
    )
    data_path = os.environ.get("CAIL_DATA", "./data/formatted_data_fixed.json")
    max_length = int(os.environ.get("QWEN_MAX_LENGTH", "1024"))
    train_limit = int(os.environ.get("QWEN_TRAIN_LIMIT", "0"))
    val_limit = int(os.environ.get("QWEN_VAL_LIMIT", "0"))
    test_limit = int(os.environ.get("QWEN_TEST_LIMIT", "0"))
    num_epochs = float(os.environ.get("QWEN_EPOCHS", "2"))
    batch_size = int(os.environ.get("QWEN_BATCH", "2"))
    grad_accum = int(os.environ.get("QWEN_GRAD_ACCUM", "8"))
    lr = float(os.environ.get("QWEN_LR", "2e-4"))
    max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
    run_tag = os.environ.get("QWEN_RUN_TAG", "").strip()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"qwen_lora_cail_{run_tag}_{timestamp}" if run_tag else f"qwen_lora_cail_{timestamp}"
    output_dir = f"./outputs/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    splits = load_split(data_path, seed=42)
    train_items = maybe_limit(splits["train"], train_limit)
    val_items = maybe_limit(splits["val"], val_limit)
    test_items = maybe_limit(splits["test"], test_limit)

    with open(os.path.join(output_dir, "split_sizes.json"), "w", encoding="utf-8") as f:
        json.dump({"train": len(train_items), "val": len(val_items), "test": len(test_items)}, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds = prepare_sft_dataset(tokenizer, train_items).map(
        tokenize_function(tokenizer, max_length),
        batched=True,
        remove_columns=["text", "prompt_text"],
    )
    val_ds = prepare_sft_dataset(tokenizer, val_items).map(
        tokenize_function(tokenizer, max_length),
        batched=True,
        remove_columns=["text", "prompt_text"],
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
        warmup_ratio=0.03,
        weight_decay=0.0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    if num_epochs > 0:
        trainer.train()
        trainer.save_model(os.path.join(output_dir, "adapter"))
        tokenizer.save_pretrained(os.path.join(output_dir, "adapter"))

    os.environ["QWEN_MAX_NEW_TOKENS"] = str(max_new_tokens)
    metrics = evaluate_generation(trainer.model, tokenizer, test_items, output_dir)
    with open(os.path.join(output_dir, "run_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Qwen LoRA CAIL Summary\n\n")
        f.write(f"- Model: {model_name}\n")
        f.write(f"- Run tag: {run_tag or 'default'}\n")
        f.write(f"- Train/Val/Test: {len(train_items)} / {len(val_items)} / {len(test_items)}\n")
        f.write(f"- Epochs: {num_epochs}\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1: {metrics['f1']:.4f}\n")


if __name__ == "__main__":
    main()
