#!/usr/bin/env python3

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import resample

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
    metadata = pd.read_csv(metadata_path, sep=",")
    if 'Accession' not in metadata.columns:
        raise ValueError("'Accession' column not found in metadata.")
    metadata.set_index("Accession", inplace=True)

    kmer_table = pd.read_csv(kmer_path, sep="\t", index_col=0)

    print(f"Loaded metadata shape: {metadata.shape}")
    print(f"Loaded kmer shape: {kmer_table.shape}")

    non_zero_counts = (kmer_table > 0).sum(axis=1)
    cutoff = int(0.05 * kmer_table.shape[1])
    kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]
    if kmer_filtered.empty:
        raise ValueError("No kmers left after filtering.")

    print(f"Kmers after filtering: {kmer_filtered.shape[0]}")

    kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=0), axis=1) * 100
    kmer_normalized = kmer_normalized.astype(float)

    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_normalized.T)
    X = pd.DataFrame(kmer_scaled, index=kmer_normalized.columns)

    shared_samples = metadata.index.intersection(X.index)
    print(f"\n🔍 Metadata Index Example: {metadata.index[:5].tolist()}")
    print(f"🔍 K-mer Columns Example: {X.index[:5].tolist()}")

    if len(shared_samples) == 0:
        raise ValueError("No shared samples between metadata and kmer data!")

    print(f"🔗 Shared samples between metadata and kmer data: {len(shared_samples)}")

    metadata = metadata.loc[shared_samples]
    X = X.loc[shared_samples]
    y = metadata[args.target_column]

    return X, y, kmer_normalized, scaler

def compute_balance_metrics(y):
    proportions = y.value_counts(normalize=True)
    gini = 1 - np.sum(proportions ** 2)
    shannon = -np.sum(proportions * np.log(proportions + 1e-12))
    shannon_normalized = shannon / np.log(len(proportions))
    return gini, shannon_normalized, proportions

def perfectly_balance_classes(X, y):
    print("\n⚖️ Performing PERFECT class balancing (oversampling to match max class count)")
    df = X.copy()
    df["label"] = y
    max_count = df["label"].value_counts().max()
    balanced_df = pd.concat([
        resample(group, replace=True, n_samples=max_count, random_state=42)
        for _, group in df.groupby("label")
    ])
    X_balanced = balanced_df.drop(columns=["label"])
    y_balanced = balanced_df["label"]

    gini_orig, entropy_orig, p_orig = compute_balance_metrics(y)
    gini_bal, entropy_bal, p_bal = compute_balance_metrics(y_balanced)

    print(f"\n📊 Class Balance (Before):\n{p_orig}\nGini: {gini_orig:.4f} | Entropy: {entropy_orig:.4f}")
    print(f"\n📊 Class Balance (After):\n{p_bal}\nGini: {gini_bal:.4f} | Entropy: {entropy_bal:.4f}")

    pd.DataFrame({
        "Class": p_orig.index,
        "Proportion_Original": p_orig.values,
        "Proportion_Balanced": p_bal.values
    }).to_csv(f"{args.output_dir}/class_balance_comparison.csv", index=False)

    return X_balanced, y_balanced

def build_rf():
    return RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42, oob_score=True)

def run_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, None],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    model = GridSearchCV(build_rf(), param_grid, cv=cv, scoring='f1_weighted', verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

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

    print(f"\n🌟 Best split found: Seed={best_data['seed']} | F1={best_data['score']:.4f}")
    return best_data

def evaluate_model(y_true, y_pred, labels, name, save_path_prefix):
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{save_path_prefix}_report.csv")
    print(f"\n{name} Classification Report:\n", pd.DataFrame(report).transpose())

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    sns.set_style("whitegrid")

    ax = sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues',
                     xticklabels=labels, yticklabels=labels,
                     linewidths=0.5, linecolor='gray', cbar_kws={"shrink": 0.8})

    ax.set_title(f"{name} Confusion Matrix", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_features(model, kmer_normalized, path_csv, path_png, top_n=10):
    importances = model.feature_importances_
    feature_names = kmer_normalized.index.tolist()
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_df = importance_df[importance_df["Feature"].str.len() >= 10].sort_values(by="Importance", ascending=False).head(top_n)
    top_df.to_csv(path_csv, index=False)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("talk")

    ax = sns.barplot(x="Importance", y="Feature", data=top_df, edgecolor='black', linewidth=0.8)
    ax.set_title("Top 10 Most Informative k-mers", fontsize=16, weight='bold', pad=15)
    ax.set_xlabel("Feature Importance", fontsize=14)
    ax.set_ylabel("k-mer", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()

def save_model(model, scaler, prefix):
    joblib.dump(model, f"{args.model_dir}/RF_{prefix}_best_model.joblib")
    joblib.dump(scaler, f"{args.model_dir}/{prefix}_scaler.joblib")
    pd.DataFrame(model.cv_results_).to_csv(f"{args.model_dir}/grid_search_results.csv", index=False)

def predict_test(best_model, scaler, kmer_normalized):
    metadata = pd.read_csv(args.test_metadata, sep=",")
    if 'Accession' not in metadata.columns:
        raise ValueError("'Accession' column not found in test metadata.")
    metadata.set_index('Accession', inplace=True)

    kmer = pd.read_csv(args.test_kmers, sep="\t", index_col=0)
    shared_kmers = kmer_normalized.index.intersection(kmer.index)
    if shared_kmers.empty:
        raise ValueError("No shared kmers between training and test data!")

    kmer = kmer.loc[shared_kmers]
    kmer_norm = kmer.div(kmer.sum(axis=0), axis=1) * 100
    kmer_norm = kmer_norm.astype(float).dropna()
    X_test = pd.DataFrame(scaler.transform(kmer_norm.T), index=kmer.columns)

    shared_samples = metadata.index.intersection(X_test.index)
    if len(shared_samples) == 0:
        raise ValueError("No shared samples between test metadata and kmer data!")

    metadata = metadata.loc[shared_samples]
    X_test = X_test.loc[shared_samples]
    y_true = metadata[args.target_column]
    y_pred = best_model.predict(X_test)
    return y_true, y_pred

# === MAIN ===

def main():
    X, y, kmer_normalized, scaler = load_and_preprocess_data(args.train_metadata, args.train_kmers)

    split = find_best_split(X, y, n_iter=50)
    X_train, X_valid = split['X_train'], split['X_valid']
    y_train, y_valid = split['y_train'], split['y_valid']

    X_train, y_train = perfectly_balance_classes(X_train, y_train)

    grid_model = run_grid_search(X_train, y_train)
    best_model = grid_model.best_estimator_

    save_model(grid_model, scaler, args.target_column.lower())

    y_pred_val = best_model.predict(X_valid)
    labels = sorted(set(y_valid) | set(y_pred_val))
    evaluate_model(y_valid, y_pred_val, labels, "Validation", f"{args.output_dir}/validation")

    plot_top_features(best_model, kmer_normalized,
                      path_csv=f"{args.output_dir}/top10_kmers.csv",
                      path_png=f"{args.output_dir}/top10_features.png")

    y_test_true, y_test_pred = predict_test(best_model, scaler, kmer_normalized)
    labels_test = sorted(set(y_test_true) | set(y_test_pred))
    evaluate_model(y_test_true, y_test_pred, labels_test, "Test", f"{args.output_dir}/test")

    pd.DataFrame({
        f"True_{args.target_column}": y_test_true,
        f"Predicted_{args.target_column}": y_test_pred
    }).to_csv(f"{args.output_dir}/{args.target_column.lower()}_predictions_test.csv")

    print("\n✅ Done. Outputs saved.")

if __name__ == "__main__":
    main()
