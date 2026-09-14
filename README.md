# Hindi Legal Document Classification (NLP)

A simple, ready-to-run pipeline for classifying Hindi legal documents
(e.g. case type, or bail granted / denied) using TF-IDF features and a
classical ML classifier (Linear SVM / Logistic Regression).

## Dataset

Primary recommendation: **HLDC — Hindi Legal Documents Corpus**
- 900K+ cleaned Hindi legal documents from Indian district courts
- Includes a ready-made bail-prediction classification task
- Repo: https://github.com/Exploration-Lab/HLDC
- Paper: https://aclanthology.org/2022.findings-acl.278/

Alternative / supplementary:
- **IL-TUR** (multi-task Indian legal benchmark, includes Hindi):
  https://huggingface.co/datasets/Exploration-Lab/IL-TUR
- **Indian Legal BERT / MILPaC** (pretrained legal embeddings for Indic languages):
  https://sites.google.com/site/saptarshighosh/datasets-codes

### Expected data format

This project expects a CSV with two columns:

```
text,label
"‌एफआईआर के अनुसार आरोपी को जमानत...", granted
"आवेदक द्वारा प्रस्तुत जमानत आवेदन खारिज...", denied
```

If you're using HLDC directly, clone their repo and follow their
`data_extraction` scripts to produce this CSV — the raw corpus ships as
per-district JSON/PDF-derived files, not a flat CSV.

## Project structure

```
hindi_legal_nlp/
├── README.md
├── requirements.txt
├── baseline_tfidf.py    # TF-IDF + LinearSVC/LogReg classifier — this IS the project
├── preprocess.py        # Hindi text cleaning utilities
└── data/
    └── sample_data.csv  # Small synthetic sample so the code runs out of the box
```

## Quick start

```bash
pip install -r requirements.txt

python baseline_tfidf.py --data data/sample_data.csv
```

Replace `data/sample_data.csv` with your real HLDC-derived CSV once prepared.
Training takes seconds even on CPU — no GPU needed.

## How it works

1. **preprocess.py** — normalizes Devanagari Unicode, strips OCR/page-number
   noise from court documents, optionally removes Hindi stopwords.
2. **baseline_tfidf.py** — converts cleaned text into TF-IDF vectors
   (unigrams + bigrams, up to 50K features), trains a Linear SVM
   (or Logistic Regression) classifier, and reports precision/recall/F1
   and a confusion matrix on a held-out test split.

## Tuning tips

- `--model logreg` to compare against Logistic Regression instead of SVM.
- Increase `ngram_range` in `baseline_tfidf.py` to `(1, 3)` if trigram legal
  phrases (e.g. "जमानत खारिज की") matter for your labels.
- For imbalanced classes (e.g. rare case types), `class_weight="balanced"`
  is already set by default.
- Legal Hindi text often mixes Devanagari with English legal terms (e.g. "FIR",
  "IPC धारा 302") — preprocessing normalizes but doesn't strip these, since
  they carry meaning.
