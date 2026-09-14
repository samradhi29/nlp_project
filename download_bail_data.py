"""
Fast baseline for Hindi legal bail outcome classification using
TF-IDF features with Linear SVM.

This version uses ONLY `facts-and-arguments` and removes
`judge-opinion` to avoid label leakage.

Usage:
    python baseline_tfidf.py --data data/train.csv --test_data data/test.csv
"""

import argparse
import ast
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import clean_text


def extract_facts(text):
    """
    Extract only facts-and-arguments from the serialized
    dictionary stored in the CSV text column.
    """
    try:
        data = ast.literal_eval(text)

        facts = data.get("facts-and-arguments", [])

        if isinstance(facts, list):
            return " ".join(str(x) for x in facts)

        return str(facts)

    except Exception:
        return ""


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.dropna(subset=[text_col, label_col])

    
    df[text_col] = df[text_col].apply(extract_facts)

    
    df[text_col] = df[text_col].apply(
        lambda t: clean_text(t, drop_stopwords=False)
    )

    
    df = df[df[text_col].str.strip() != ""]

    return df


def build_pipeline(model_name: str = "svm") -> Pipeline:

    # Word n-grams capture useful legal phrases such as:
    # "जमानत प्रार्थना", "जमानत खारिज", etc.
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )

    if model_name == "svm":
        clf = LinearSVC(
            class_weight="balanced",
            C=1.0
        )

    elif model_name == "logreg":
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0
        )

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="Path to training CSV"
    )

    parser.add_argument(
        "--test_data",
        default=None,
        help="Optional separate test CSV"
    )

    parser.add_argument(
        "--text_col",
        default="text"
    )

    parser.add_argument(
        "--label_col",
        default="label"
    )

    parser.add_argument(
        "--model",
        default="svm",
        choices=["svm", "logreg"]
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    parser.add_argument(
        "--out",
        default="model_tfidf_facts_only.joblib"
    )

    args = parser.parse_args()

   
    if args.test_data:

        train_df = load_data(
            args.data,
            args.text_col,
            args.label_col
        )

        test_df = load_data(
            args.test_data,
            args.text_col,
            args.label_col
        )

        print(
            f"Loaded {len(train_df)} train docs, "
            f"{len(test_df)} test docs, "
            f"{train_df[args.label_col].nunique()} classes"
        )

        X_train = train_df[args.text_col]
        y_train = train_df[args.label_col]

        X_test = test_df[args.text_col]
        y_test = test_df[args.label_col]

    else:

        df = load_data(
            args.data,
            args.text_col,
            args.label_col
        )

        print(
            f"Loaded {len(df)} documents, "
            f"{df[args.label_col].nunique()} classes"
        )

        X_train, X_test, y_train, y_test = train_test_split(
            df[args.text_col],
            df[args.label_col],
            test_size=args.test_size,
            random_state=42,
            stratify=df[args.label_col],
        )

    

    pipeline = build_pipeline(args.model)

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    

    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    

    joblib.dump(pipeline, args.out)

    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()