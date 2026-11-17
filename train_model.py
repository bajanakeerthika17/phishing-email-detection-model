"""
train_model.py
--------------
Complete training and evaluation pipeline for Phishing Email Detection.
Trains, evaluates, and compares three core machine learning algorithms:
1. Logistic Regression
2. Multinomial Naive Bayes
3. Random Forest Classifier

Outputs:
- models/phishing_model.pkl (Best model bundle with vectorizer & metrics)
- models/model_metrics.json (Evaluation scores for all 3 models)
- images/confusion_matrix.png (High-res confusion matrix heatmap)
- images/model_comparison.png (Algorithm performance comparison chart)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

# Ensure custom writable Matplotlib directory to suppress warnings on Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_DIR = os.path.join(BASE_DIR, ".mplconfig")
os.makedirs(MPL_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = MPL_DIR

# Headless backend for Matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import clean_dataset, clean_text, get_tfidf_vectorizer

MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


def load_and_preprocess_data():
    """
    Loads dataset, handles null values and duplicates, and prepares data.
    """
    primary_dataset = os.path.join(BASE_DIR, "phishing_email_dataset.csv")
    fallback_dataset = os.path.join(BASE_DIR, "dataset.csv")

    if os.path.exists(primary_dataset):
        df_raw = pd.read_csv(primary_dataset)
        dataset_source = primary_dataset
    elif os.path.exists(fallback_dataset):
        df_raw = pd.read_csv(fallback_dataset)
        dataset_source = fallback_dataset
    else:
        print("[!] No dataset file found. Generating dataset now...")
        from create_dataset import build_dataset
        build_dataset()
        df_raw = pd.read_csv(primary_dataset)
        dataset_source = primary_dataset

    print(f"\n[Step 1] Loading Dataset from: {os.path.basename(dataset_source)}")
    # Clean dataset (drop missing and duplicates)
    df = clean_dataset(df_raw, text_col="email_text", label_col="label")

    print(f"         Total valid records: {len(df)}")
    print(f"         Phishing emails (1): {(df['label'] == 1).sum()}")
    print(f"         Safe emails (0):     {(df['label'] == 0).sum()}")

    # Apply text cleaning
    print("\n[Step 2] Preprocessing Email Text (Lowercase, token cleaning, stopwords)...")
    df["cleaned_text"] = df["email_text"].apply(lambda t: clean_text(t, remove_stopwords=False))

    return df


def train_and_evaluate():
    """
    Orchestrates data splitting, TF-IDF vectorization, model training,
    evaluation, model comparison, visualization, and artifact export.
    """
    print("=" * 70)
    print("      PHISHING EMAIL DETECTION - MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    df = load_and_preprocess_data()

    X = df["cleaned_text"]
    y = df["label"]

    # 2. Train-Test Split (Avoiding Data Leakage)
    # 80% Training, 20% Testing, Stratified by label
    print("\n[Step 3] Splitting Dataset (80% Train, 20% Test, Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"         Training set size: {len(X_train)} samples")
    print(f"         Testing set size:  {len(X_test)} samples")

    # 3. Text Vectorization with TF-IDF
    # Fit ONLY on X_train to prevent data leakage!
    print("\n[Step 4] Extracting Features via TF-IDF (Term Frequency-Inverse Document Frequency)...")
    vectorizer = get_tfidf_vectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"         Vocabulary size: {vocab_size} n-grams")
    print(f"         Training feature matrix: {X_train_tfidf.shape}")

    # 4. Define the 3 Classifiers with tuned parameters
    models = {
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(alpha=0.2),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_features="log2", class_weight="balanced", random_state=42)
    }

    results = {}
    fitted_models = {}

    print("\n[Step 5] Training and Evaluating 3 Classification Models...")
    print("-" * 70)

    for name, clf in models.items():
        print(f"\n--> Training: {name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"], output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, target_names=["Safe", "Phishing"], zero_division=0)

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "classification_report_text": report_text
        }
        fitted_models[name] = clf

        print(f"    * Accuracy:   {acc * 100:.2f}%")
        print(f"    * Precision:  {prec * 100:.2f}%")
        print(f"    * Recall:     {rec * 100:.2f}%")
        print(f"    * F1-Score:   {f1 * 100:.2f}%")

    # 5. Select Best Model based on F1-Score (and Accuracy as secondary tie-breaker)
    best_model_name = max(results.keys(), key=lambda k: (results[k]["f1_score"], results[k]["accuracy"]))
    best_model = fitted_models[best_model_name]
    best_metrics = results[best_model_name]

    print("\n" + "=" * 70)
    print(f"[BEST MODEL SELECTED]: {best_model_name.upper()}")
    print(f"F1-Score: {best_metrics['f1_score'] * 100:.2f}% | Accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    print("=" * 70)
    print("\nDetailed Classification Report for Best Model:")
    print(best_metrics["classification_report_text"])

    # 6. Generate Evaluation Visualizations
    print("[Step 6] Generating Evaluation Visualizations (Matplotlib & Seaborn)...")
    generate_confusion_matrix_plot(best_model_name, best_metrics["confusion_matrix"])
    generate_comparison_chart(results)

    # 7. Save Model Artifacts
    print("\n[Step 7] Saving Model Artifacts...")
    
    # Save the bundle to models/phishing_model.pkl (Requested format)
    model_bundle = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "vectorizer": vectorizer,
        "results": results,
        "all_models": fitted_models,
        "feature_count": vocab_size
    }
    bundle_path = os.path.join(MODELS_DIR, "phishing_model.pkl")
    joblib.dump(model_bundle, bundle_path)
    print(f"         Saved primary model bundle to: {bundle_path}")

    # Also save standard files in root for direct backwards compatibility
    joblib.dump(best_model, os.path.join(BASE_DIR, "phishing_model.joblib"))
    joblib.dump(vectorizer, os.path.join(BASE_DIR, "tfidf_vectorizer.joblib"))

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"         Saved model metrics to: {metrics_path}")

    # Root metrics fallback
    with open(os.path.join(BASE_DIR, "model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print("         TRAINING PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 70)
    return model_bundle


def generate_confusion_matrix_plot(model_name: str, cm_list: list):
    """
    Creates and saves a styled Seaborn heatmap of the confusion matrix.
    """
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    # Labels and annotations
    group_names = ["True Safe (TN)", "False Phishing (FP)", "False Safe (FN)", "True Phishing (TP)"]
    group_counts = [f"{value:0.0f}" for value in cm.flatten()]
    group_percentages = [f"{value:.1%}" for value in cm.flatten() / np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Safe", "Predicted Phishing"],
        yticklabels=["Actual Safe", "Actual Phishing"],
        linewidths=1.5,
        linecolor="#E2E8F0",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax
    )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=15, color="#1E293B")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", labelpad=10, color="#334155")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", labelpad=10, color="#334155")
    plt.tight_layout()

    out_path = os.path.join(IMAGES_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"         Generated Confusion Matrix plot at: {out_path}")


def generate_comparison_chart(results: dict):
    """
    Creates and saves a grouped bar chart comparing the performance metrics
    of Logistic Regression, Naive Bayes, and Random Forest.
    """
    models = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

    data = {m: [results[m][metric] * 100 for metric in metrics_names] for m in models}

    x = np.arange(len(display_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    colors = ["#2563EB", "#059669", "#7C3AED"]

    for i, (model_name, scores) in enumerate(data.items()):
        rects = ax.bar(x + (i - 1) * width, scores, width, label=model_name, color=colors[i % len(colors)], alpha=0.9, edgecolor="white", linewidth=1.2)
        # Add score labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold"
            )

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold", color="#334155")
    ax.set_title("Machine Learning Algorithm Comparison", fontsize=13, fontweight="bold", pad=15, color="#1E293B")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11, fontweight="bold", color="#334155")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(IMAGES_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"         Generated Algorithm Comparison chart at: {out_path}")


if __name__ == "__main__":
    train_and_evaluate()
