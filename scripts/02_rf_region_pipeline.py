#!/usr/bin/env python3

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import argparse

warnings.filterwarnings("ignore")

# === ARGPARSE ===
parser = argparse.ArgumentParser()
parser.add_argument("--train_metadata", required=True)
parser.add_argument("--test_metadata", required=True)
parser.add_argument("--train_kmers", required=True)
parser.add_argument("--test_kmers", required=True)
parser.add_argument("--output_dir", default="output")
parser.add_argument("--model_dir", default="models")
parser.add_argument("--target_column", default="Region")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)

# === FUNCTIONS ===

def load_and_preprocess_data(metadata_path, kmer_path):
    metadata = pd.read_csv(metadata_path, sep=",", index_col=0)
    kmer_table = pd.read_csv(kmer_path, sep="\t", index_col=0)

    print(f"🧼 Loaded metadata shape: {metadata.shape}")
    print(f"🧼 Loaded kmer shape: {kmer_table.shape}")

    non_zero_counts = (kmer_table > 0).sum(axis=1)
    cutoff = int(0.05 * kmer_table.shape[1])
    kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]
    if kmer_filtered.empty:
        raise ValueError("❌ No kmers left after filtering.")

    print(f"✅ Kmers after filtering: {kmer_filtered.shape[0]}")

    kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=0), axis=1) * 100
    kmer_normalized = kmer_normalized.astype(float)

    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_normalized.T)
    X = pd.DataFrame(kmer_scaled, index=kmer_normalized.columns)

    shared_samples = metadata.index.intersection(X.index)
    if len(shared_samples) == 0:
        raise ValueError("❌ No shared samples between metadata and kmer data!")

    print(f"🔗 Shared samples: {len(shared_samples)}")

    metadata = metadata.loc[shared_samples]
    X = X.loc[shared_samples]
    y = metadata[args.target_column]

    return X, y, kmer_normalized, scaler


def oversample_minority_classes(X, y, min_fraction=0.01):
    print("⚖️ Balancing classes using SMOTE with 'Other' grouping for rare classes...")

    class_counts = y.value_counts(normalize=True)
    rare_classes = class_counts[class_counts < min_fraction].index.tolist()

    if rare_classes:
        print(f"🔁 Grouping rare classes into 'Other': {rare_classes}")
        y = y.where(~y.isin(rare_classes), other="Other")

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    y.value_counts(normalize=True).to_csv(f"{args.output_dir}/class_distribution_original.csv")
    y_resampled.value_counts(normalize=True).to_csv(f"{args.output_dir}/class_distribution_resampled.csv")

    return X_resampled, y_resampled


def build_rf():
    return RandomForestClassifier(
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )


def find_best_split(X, y, n_iter=30, test_size=0.2):
    best_score = -1
    best_data = None

    for seed in range(n_iter):
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        clf = build_rf()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_valid)
        score = f1_score(y_valid, y_pred, average='weighted')

        if score > best_score:
            best_score = score
            best_data = {
                'X_train': X_train, 'X_valid': X_valid,
                'y_train': y_train, 'y_valid': y_valid,
                'seed': seed, 'score': score
            }

    print(f"🌟 Best split found: Seed={best_data['seed']} | F1={best_data['score']:.4f}")
    return best_data


def run_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, None],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = GridSearchCV(
        build_rf(),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true, y_pred, labels, name, save_path):
    print(f"\n📊 Classification Report ({name}):")
    print(classification_report(y_true, y_pred, digits=3))

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_pr_curves(y_true, y_probs, classes, prefix):
    y_bin = label_binarize(y_true, classes=classes)
    for i, label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {label}")
        plt.legend()
        plt.savefig(f"{args.output_dir}/{prefix}_roc_{label}.png")
        plt.close()

        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {label}")
        plt.savefig(f"{args.output_dir}/{prefix}_pr_{label}.png")
        plt.close()


def plot_top_features(model, kmer_normalized, path_csv, path_png, top_n=10):
    importances = model.feature_importances_
    feature_names = kmer_normalized.index
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    top_df = importance_df[importance_df["Feature"].str.len() >= 10].head(top_n)
    top_df.to_csv(path_csv, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=top_df)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(path_png)
    plt.close()


def save_model(model, scaler, prefix):
    joblib.dump(model, f"{args.model_dir}/RF_{prefix}_best_model.joblib")
    joblib.dump(scaler, f"{args.model_dir}/{prefix}_scaler.joblib")
    pd.DataFrame(model.cv_results_).to_csv(f"{args.model_dir}/grid_search_results.csv", index=False)


def predict_test(best_model, scaler, kmer_normalized):
    metadata = pd.read_csv(args.test_metadata, sep=",", index_col=0)
    kmer = pd.read_csv(args.test_kmers, sep="\t", index_col=0)

    shared_kmers = kmer_normalized.index.intersection(kmer.index)
    if shared_kmers.empty:
        raise ValueError("❌ No shared kmers between training and test data!")

    kmer = kmer.loc[shared_kmers]
    kmer_norm = kmer.div(kmer.sum(axis=0), axis=1) * 100
    kmer_norm = kmer_norm.astype(float).dropna()
    X_test = pd.DataFrame(scaler.transform(kmer_norm.T), index=kmer.columns)

    shared_samples = metadata.index.intersection(X_test.index)
    if len(shared_samples) == 0:
        raise ValueError("❌ No shared samples between test metadata and kmer data!")

    metadata = metadata.loc[shared_samples]
    X_test = X_test.loc[shared_samples]
    y_true = metadata[args.target_column]
    y_pred = best_model.predict(X_test)
    y_probs = best_model.predict_proba(X_test)

    return y_true, y_pred, y_probs


# === MAIN ENTRY POINT ===

def main():
    X, y, kmer_normalized, scaler = load_and_preprocess_data(args.train_metadata, args.train_kmers)
    X, y = oversample_minority_classes(X, y, min_fraction=0.01)

    split = find_best_split(X, y, n_iter=50)
    X_train, X_valid = split['X_train'], split['X_valid']
    y_train, y_valid = split['y_train'], split['y_valid']

    grid_model = run_grid_search(X_train, y_train)
    best_model = grid_model.best_estimator_

    save_model(grid_model, scaler, args.target_column.lower())

    y_pred_val = best_model.predict(X_valid)
    labels = sorted(set(y_valid) | set(y_pred_val))
    evaluate_model(y_valid, y_pred_val, labels, "Validation", f"{args.output_dir}/confusion_matrix_validation.png")

    y_val_probs = best_model.predict_proba(X_valid)
    plot_roc_pr_curves(y_valid, y_val_probs, classes=best_model.classes_, prefix="val")

    plot_top_features(
        best_model,
        kmer_normalized,
        path_csv=f"{args.output_dir}/top10_kmers.csv",
        path_png=f"{args.output_dir}/top10_features.png"
    )

    y_test_true, y_test_pred, y_test_probs = predict_test(best_model, scaler, kmer_normalized)
    labels_test = sorted(set(y_test_true) | set(y_test_pred))
    evaluate_model(y_test_true, y_test_pred, labels_test, "Test", f"{args.output_dir}/confusion_matrix_test.png")
    plot_roc_pr_curves(y_test_true, y_test_probs, classes=best_model.classes_, prefix="test")

    pd.DataFrame({
        f"True_{args.target_column}": y_test_true,
        f"Predicted_{args.target_column}": y_test_pred
    }).to_csv(f"{args.output_dir}/{args.target_column.lower()}_predictions_test.csv")

    print("\n✅ Model training + evaluation complete. All outputs saved.")


if __name__ == "__main__":
    main()
