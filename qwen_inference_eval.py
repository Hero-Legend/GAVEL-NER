import datetime
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_lora_cail_eval import (
    build_generation_prompt,
    maybe_limit,
    normalize_gold_entities,
    parse_prediction,
    load_split,
    set_seed,
)


def load_model_and_tokenizer(base_model: str, adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def strict_eval_with_latency(model, tokenizer, test_items, max_new_tokens: int):
    device = next(model.parameters()).device
    true_positives = 0
    pred_positives = 0
    actual_positives = 0
    latencies_ms = []
    samples = []

    with torch.no_grad():
        for idx, item in enumerate(test_items):
            text = item.get("context") or item.get("text") or ""
            prompt = build_generation_prompt(tokenizer, text)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

            completion = tokenizer.decode(
                generated[0][inputs["input_ids"].shape[1]:],
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
                        "latency_ms": latencies_ms[-1],
                        "raw_output": completion,
                        "parsed_json": raw_json,
                    }
                )

    precision = true_positives / pred_positives if pred_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    lat_sorted = sorted(latencies_ms)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p90 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.9))] if lat_sorted else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "pred_positives": pred_positives,
        "actual_positives": actual_positives,
        "test_size": len(test_items),
        "latency_ms_mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "gpu_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        "samples": samples,
    }


def main():
    set_seed(42)
    base_model = os.environ.get("QWEN_MODEL", "/gz-fs/storage/models/Qwen2.5-7B-Instruct")
    adapter_path = os.environ.get("QWEN_ADAPTER", "").strip() or None
    data_path = os.environ.get("CAIL_DATA", "./data/formatted_data_fixed.json")
    test_limit = int(os.environ.get("QWEN_TEST_LIMIT", "50"))
    max_new_tokens = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
    run_tag = os.environ.get("QWEN_RUN_TAG", "inference")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/qwen_inference_eval_{run_tag}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    splits = load_split(data_path, seed=42)
    test_items = maybe_limit(splits["test"], test_limit)

    model, tokenizer = load_model_and_tokenizer(base_model, adapter_path)
    metrics = strict_eval_with_latency(model, tokenizer, test_items, max_new_tokens)

    samples = metrics.pop("samples")
    metrics["base_model"] = base_model
    metrics["adapter_path"] = adapter_path

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "sample_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Qwen Inference Evaluation\n\n")
        f.write(f"- Base model: {base_model}\n")
        f.write(f"- Adapter: {adapter_path or 'none'}\n")
        f.write(f"- Test size: {metrics['test_size']}\n")
        f.write(f"- Precision: {metrics['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['recall']:.4f}\n")
        f.write(f"- F1: {metrics['f1']:.4f}\n")
        f.write(f"- Mean latency (ms): {metrics['latency_ms_mean']:.2f}\n")
        f.write(f"- P50 latency (ms): {metrics['latency_ms_p50']:.2f}\n")
        f.write(f"- P90 latency (ms): {metrics['latency_ms_p90']:.2f}\n")
        f.write(f"- Max CUDA memory allocated (MB): {metrics['gpu_memory_allocated_mb']:.2f}\n")


if __name__ == "__main__":
    main()
