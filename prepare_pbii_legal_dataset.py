import argparse
import json
from collections import Counter
from pathlib import Path


def repair_mojibake(text: str) -> str:
    if not text:
        return text
    try:
        return text.encode("gb18030", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return text


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def normalize_record(record):
    text = [repair_mojibake(tok) for tok in record["text"]]
    labels = record["labels"]
    if len(text) != len(labels):
        raise ValueError(f"Length mismatch in record {record.get('id')}: {len(text)} vs {len(labels)}")
    return {
        "id": record.get("id"),
        "text": text,
        "labels": labels,
    }


def write_bio(records, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            for ch, tag in zip(rec["text"], rec["labels"]):
                f.write(f"{ch}\t{tag}\n")
            f.write("\n")


def write_jsonl(records, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(records):
    label_counter = Counter()
    entity_counter = Counter()
    lengths = []
    for rec in records:
        lengths.append(len(rec["text"]))
        for tag in rec["labels"]:
            label_counter[tag] += 1
            if tag.startswith("B-"):
                entity_counter[tag[2:]] += 1
    lengths_sorted = sorted(lengths)
    p95_index = max(0, int(len(lengths_sorted) * 0.95) - 1)
    return {
        "num_samples": len(records),
        "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "p95_length": lengths_sorted[p95_index] if lengths else 0,
        "entity_starts": dict(sorted(entity_counter.items())),
        "label_counts": dict(sorted(label_counter.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare the PBII legal NER dataset for GAVEL experiments.")
    parser.add_argument("--input-dir", required=True, help="Directory containing train.txt/dev.txt/labels.txt/train1.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for converted PBII files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = [normalize_record(r) for r in load_jsonl(input_dir / "train.txt")]
    dev_records = [normalize_record(r) for r in load_jsonl(input_dir / "dev.txt")]

    write_bio(train_records, output_dir / "train_bio")
    write_bio(dev_records, output_dir / "test_bio")
    write_jsonl(train_records, output_dir / "train_fixed.jsonl")
    write_jsonl(dev_records, output_dir / "dev_fixed.jsonl")

    raw_labels = (input_dir / "labels.txt").read_text(encoding="utf-8", errors="ignore").splitlines()
    labels = [repair_mojibake(x).strip() for x in raw_labels if x.strip()]
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    summary = {
        "dataset": "PBII legal NER",
        "source_files": {
            "train": str(input_dir / "train.txt"),
            "dev": str(input_dir / "dev.txt"),
            "labels": str(input_dir / "labels.txt"),
            "raw_json": str(input_dir / "train1.json"),
        },
        "labels": labels,
        "train": summarize(train_records),
        "dev": summarize(dev_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = [
        "# PBII Preparation Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Labels: `{', '.join(labels)}`",
        f"- Train samples: `{summary['train']['num_samples']}`",
        f"- Dev samples: `{summary['dev']['num_samples']}`",
        f"- Train avg length: `{summary['train']['avg_length']}`",
        f"- Dev avg length: `{summary['dev']['avg_length']}`",
        f"- Train max length: `{summary['train']['max_length']}`",
        f"- Dev max length: `{summary['dev']['max_length']}`",
        "",
        "Outputs:",
        f"- `{output_dir / 'train_bio'}`",
        f"- `{output_dir / 'test_bio'}`",
        f"- `{output_dir / 'train_fixed.jsonl'}`",
        f"- `{output_dir / 'dev_fixed.jsonl'}`",
        f"- `{output_dir / 'summary.json'}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
