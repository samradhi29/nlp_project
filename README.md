# Hindi Legal Bail Outcome Classification

## Problem Statement

Hindi legal documents contain large amounts of unstructured text, making manual analysis time-consuming. This project focuses on automatically predicting the outcome of bail cases from Hindi legal documents.

The objective is to predict whether bail is **Granted** or **Denied** using only the **Facts and Arguments** of a case.

The `judge-opinion` section is excluded from the input to avoid **label leakage**, since it contains the judge's reasoning related to the final decision.

---

## Dataset

This project uses the **BAIL task from IL-TUR (Indian Legal Text Understanding and Reasoning)**, based on the **Hindi Legal Documents Corpus (HLDC)**.

**Dataset:**
https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/bail/train_all?p=1

The dataset contains Hindi legal documents with sections including:

* `facts-and-arguments`
* `judge-opinion`
* Bail outcome label

The classification labels are:

* `GRANTED`
* `DENIED`

### Input and Output

```text
Facts and Arguments
        |
        v
   NLP Processing
        |
        v
   TF-IDF Features
        |
        v
   Linear SVM
        |
        v
Bail Granted / Bail Denied
```

---

## Reference Paper

This project is based on the bail prediction task introduced in:

**HLDC: Hindi Legal Documents Corpus**

The paper introduced a large-scale Hindi legal corpus containing **912,568 legal documents** and explored bail outcome prediction as a downstream task.

**Paper:**
https://arxiv.org/abs/2204.00806

**Official Repository:**
https://github.com/Exploration-Lab/HLDC

The original paper experimented with:

* Doc2Vec + SVM
* Doc2Vec + XGBoost
* IndicBERT
* TF-IDF + IndicBERT
* TextRank + IndicBERT
* Salience Prediction + IndicBERT
* Multi-Task Learning

This project implements a **classical NLP baseline** for the same bail prediction task using TF-IDF and Linear SVM.

---

## Methodology

### 1. Facts and Arguments Extraction

The raw document contains multiple sections.

Only the following section is used:

```text
facts-and-arguments
```

The following section is excluded:

```text
judge-opinion
```

This prevents the model from directly learning information from the judge's reasoning.

---

### 2. Text Preprocessing

The extracted Hindi text is cleaned using:

* Unicode NFC normalization
* Noise removal
* Whitespace normalization

Stopword removal is currently disabled.

---

### 3. TF-IDF Feature Extraction

The cleaned Hindi text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Config
