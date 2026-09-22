# Hindi Legal Bail Outcome Classification

## Problem Statement

Our project focuses on using Natural Language Processing and Machine Learning to classify Hindi legal bail documents as either GRANTED or DENIED. Since legal documents contain large amounts of information, manually analyzing them can be time-consuming. We aim to automatically identify patterns from the facts and arguments presented in the case and use them to predict the bail outcome. However, we deliberately exclude the judge’s opinion and final decision because including them could cause data leakage, allowing the model to simply learn the answer rather than understand the underlying case information

---

## Dataset

The project uses the **BAIL task from IL-TUR**, based on the **Hindi Legal Documents Corpus (HLDC)**.

**Dataset:**
https://huggingface.co/datasets/Exploration-Lab/IL-TUR/viewer/bail/train_all?p=1

The Hindi Legal Documents Corpus (HLDC) is a large Hindi legal dataset created from district court documents of Uttar Pradesh, India. The original corpus contains 912,568 Hindi legal documents, collected from the e-Courts website for cases from May 2019 to May 2021.

For the bail classification task, the researchers extracted 340,280 bail-related documents from HLDC. These documents were structured into sections such as facts and arguments and judge opinion, with the final bail decision labelled as GRANTED or DENIED.

---
For reference, we are using the **HLDC (Hindi Legal Documents Corpus)** research paper as our base paper. It contains data collected from the Indian e-Courts system and presents a solution approach that we have taken as a reference to follow in our project.

One of the main reasons for choosing this direction is that we found the use of AI in the field of the judiciary is not as widespread as it should be compared to other domains. While exploring different areas, we noticed a lack of AI-based solutions and research focused on supporting the Indian judicial system.

Therefore, we decided to explore this direction and understand how AI can be used as a supporting tool for legal document analysis and decision assistance. We are following the ideas and methodology proposed in the HLDC research paper as the foundation for our work.
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

