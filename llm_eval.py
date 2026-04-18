import json
import os
import random
import concurrent.futures

import torch
from openai import OpenAI
from torch.utils.data import random_split
from tqdm import tqdm

LLM_CONFIGS = {
    "DeepSeek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "Qwen": {
        "api_key_env": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-max",
    },
    "Kimi": {
        "api_key_env": "KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
    },
}

VALID_LABELS = {
    "NHVI",
    "NHCS",
    "NCSP",
    "NCSM",
    "NCGV",
    "NT",
    "NS",
    "NASI",
    "NATS",
    "NO",
}

LABEL_DESCRIPTIONS = {
    "NHVI": "victim",
    "NHCS": "suspect or defendant",
    "NCSP": "public security, procuratorate, or court institution",
    "NCSM": "forensic, appraisal, or medical institution",
    "NCGV": "other government administrative institution",
    "NT": "time expression",
    "NS": "location",
    "NASI": "stolen money or stolen property",
    "NATS": "crime tool",
    "NO": "other case-related item",
}


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)



def load_test_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_data = []
    for item in data:
        text = item.get("context") or item.get("text") or ""
        if text:
            valid_data.append(item)

    set_seed(42)
    indices = list(range(len(valid_data)))
    train_size = int(0.8 * len(indices))
    val_size = int(0.1 * len(indices))
    test_size = len(indices) - train_size - val_size
    _, _, test_indices = random_split(indices, [train_size, val_size, test_size])
    return [valid_data[i] for i in test_indices]



def get_client(config):
    api_key = os.getenv(config["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Please set environment variable {config['api_key_env']}."
        )
    return OpenAI(api_key=api_key, base_url=config["base_url"])



def build_prompt(text):
    label_block = "\n".join(
        f"- {code}: {description}" for code, description in LABEL_DESCRIPTIONS.items()
    )
    return f"""You are an information extraction assistant for Chinese judicial texts.
Extract named entities from the input text.

Allowed labels:
{label_block}

Return a JSON array only.
Each item must have this schema:
{{"entity": "entity surface form", "type": "one allowed label code"}}

If no entity is found, return [].
Do not return markdown or extra commentary.

Text:
{text}
"""



def normalize_predictions(raw_content):
    try:
        start = raw_content.find("[")
        end = raw_content.rfind("]")
        if start == -1 or end == -1:
            return set()
        entities = json.loads(raw_content[start : end + 1])
    except Exception:
        return set()

    extracted = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        entity = str(ent.get("entity", "")).strip()
        entity_type = str(ent.get("type", "")).strip().upper()
        if entity and entity_type in VALID_LABELS:
            extracted.add((entity, entity_type))
    return extracted



def call_llm(text, config):
    client = get_client(config)
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": build_prompt(text)},
        ],
        temperature=0.1,
    )
    raw_content = response.choices[0].message.content or "[]"
    return normalize_predictions(raw_content)



def extract_gold_entities(item):
    gold = set()
    for entity in item.get("entities", []):
        entity_text = str(entity.get("entity", "")).strip()
        entity_type = str(entity.get("label") or entity.get("type") or "").strip()
        if entity_text and entity_type:
            gold.add((entity_text, entity_type))
    return gold



def evaluate_model(model_name, test_data):
    config = LLM_CONFIGS[model_name]
    print(f"\nEvaluating {model_name} ({config['model']})")

    true_positives = 0
    pred_positives = 0
    actual_positives = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for item in test_data:
            text = item.get("context") or item.get("text") or ""
            futures.append((item, executor.submit(call_llm, text, config)))

        for item, future in tqdm(futures, total=len(futures), desc=model_name):
            predicted_entities = future.result()
            true_entities = extract_gold_entities(item)

            true_positives += len(predicted_entities.intersection(true_entities))
            pred_positives += len(predicted_entities)
            actual_positives += len(true_entities)

    precision = true_positives / pred_positives if pred_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(
        f"{model_name} - Precision: {precision * 100:.2f}%, "
        f"Recall: {recall * 100:.2f}%, F1: {f1 * 100:.2f}%"
    )
    return precision, recall, f1


if __name__ == "__main__":
    data_path = "./data/formatted_data_fixed.json"
    test_data = load_test_data(data_path)
    print(f"Loaded {len(test_data)} test samples.")

    for model_name in ["DeepSeek", "Qwen", "Kimi"]:
        evaluate_model(model_name, test_data)
