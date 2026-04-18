# GAVEL-NER

GAVEL-NER is a deployment-oriented Chinese legal named entity recognition project for strict exact-match extraction, low-resource judicial settings, and memory-bounded local deployment.

This repository contains the public code and dataset assets used for the GAVEL-NER experiments.

## Method summary

GAVEL-NER combines:

- adaptive gated fusion over lexical boundary and POS priors;
- token-level supervised contrastive alignment for long-tailed legal entities;
- CRF-free linear decoding for deployment-friendly inference.

## Repository note

This public repository is the legacy code release for the project formerly organized under the `MVCL-NER` label. The current paper and public method name should be treated as `GAVEL-NER`.

## What is included

- `data/`: dataset files and preprocessing assets
- `ablation1.py` to `ablation4.py`: ablation and variant experiment scripts
- `MVCL_bert.py`: core training and evaluation implementation retained under its legacy filename for compatibility
- `llm_eval.py`: generative baseline evaluation utilities

## Recommended citation label

If you refer to this project in manuscripts, slides, or system descriptions, please use the name `GAVEL-NER`.