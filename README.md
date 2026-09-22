# Hindi Legal Bail Outcome Classification

## Problem Statement

This project predicts the outcome of Hindi bail cases as **GRANTED** or **DENIED** using only the **Facts and Arguments** section of legal documents.

The `judge-opinion` section is excluded to avoid **label leakage**, as it may contain reasoning directly related to the final decision.

---

## Dataset

The project uses the **BAIL task from IL-TUR**, based on the **Hindi Legal Documents Corpus (HLDC)**.

**Dataset:**
https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/bail/train_all?p=1

The Hindi Legal Documents Corpus (HLDC) is a large Hindi legal dataset created from district court documents of Uttar Pradesh, India. The original corpus contains 912,568 Hindi legal documents, collected from the e-Courts website for cases from May 2019 to May 2021.

For the bail classification task, the researchers extracted 340,280 bail-related documents from HLDC. These documents were structured into sections such as facts and arguments and judge opinion, with the final bail decision labelled as GRANTED or DENIED.

---

## Pipeline

```text
IL-TUR / HLDC Dataset
        ↓
Extract Facts & Arguments
        ↓
Remove Missing / Invalid Text
        ↓
Unicode Normalization
        ↓
Whitespace / Noise Cleaning
        ↓
Train / Validation / Test Split
        ↓
TF-IDF Vectorization
        ↓
Linear SVM
        ↓
Prediction
        ↓
GRANTED / DENIED
        ↓
Accuracy, Precision, Recall, F1
```

### 1. Text Extraction

Only `facts-and-arguments` is used as model input.

`judge-opinion` is excluded to prevent label leakage.

### 2. Preprocessing

* Unicode NFC normalization
* Noise removal
* Whitespace normalization
* Stopword removal disabled

### 3. TF-IDF

The cleaned Hindi text is converted into numerical features using **TF-IDF**. The vectorizer is fitted only on the training data and then used to transform validation and test data.

### 4. Classification

A **Linear SVM (LinearSVC)** is trained on the TF-IDF features to classify cases into:

```text
GRANTED
DENIED
```

### 5. Evaluation

The model is evaluated on unseen test data using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## Model

```text
TF-IDF + Linear SVM
```

This provides a lightweight classical NLP baseline for Hindi legal text classification.

---

## Reference Paper

**HLDC: Hindi Legal Documents Corpus**

Paper: https://arxiv.org/abs/2204.00806

Repository: https://github.com/Exploration-Lab/HLDC

The original work explored approaches including Doc2Vec, SVM, XGBoost, IndicBERT, and multi-task learning. This project implements a simpler **TF-IDF + Linear SVM baseline**.

