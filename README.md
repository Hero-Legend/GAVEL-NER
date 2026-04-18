# GAVEL-NER

GAVEL-NER is a deployment-oriented Chinese legal named entity recognition project for strict exact-match extraction, low-resource judicial settings, and memory-bounded local deployment.

This public release tracks the current paper-facing experiment code rather than the earliest MVCL-NER prototype snapshot. Legacy class names such as `MVCLBert` or `MVCLDataset` are retained in some scripts for compatibility, but the public project and manuscript name is `GAVEL-NER`.

## Included in this release

- current CAIL2021 experiment scripts for CRF-free vs CRF comparison, latency, noise robustness, learning curves, and ablations
- span-based and cross-dataset comparison scripts used in the current revision cycle
- Qwen adaptation and inference evaluation scripts
- CAIL2021 formatted data under `data/formatted_data_fixed.json`
- PBII legal-domain data under `data/pbii_raw/` and prepared PBII files under `data/pbii/`

## Main entry points

- `cail_multiseed_crf_compare.py`: CRF-free vs CRF comparison on the fixed CAIL split
- `modern_span_baselines.py`: span-based baselines on the same CAIL split
- `noise_robustness_eval.py`: robustness against corrupted BMES/POS features
- `latency_breakdown.py` and `latency_breakdown_crf.py`: deployment latency profiling
- `cail_learning_curve.py`: low-resource learning-curve study
- `cross_dataset_bio.py`: PBII and other BIO-format cross-dataset evaluation
- `prepare_pbii_legal_dataset.py`: converts raw PBII files into the BIO/jsonl format used by `cross_dataset_bio.py`
- `qwen_lora_cail_eval.py` and `qwen_inference_eval.py`: adapted and prompted LLM evaluation

## Data layout

- `data/formatted_data_fixed.json`: formatted CAIL2021 data used by the main scripts in this repository
- `data/raw/xxcq_mid.jsonl`: raw jsonl snapshot retained for reference
- `data/pbii_raw/`: original PBII release files available in the local workspace
- `data/pbii/`: prepared PBII files for direct use with `cross_dataset_bio.py`

## Setup notes

Most scripts expect a local RoBERTa backbone at `./model_path/chinese-roberta-wwm-ext`. Model checkpoints and output folders are intentionally not tracked in this public repository.

`llm_eval.py` is not part of this curated release because the current paper revision uses `qwen_lora_cail_eval.py` and `qwen_inference_eval.py` instead of the earliest prototype script.
