#!/usr/bin/env python3

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# === GLOBAL SETTINGS ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === FUNCTIONS ===

def load_and_preprocess_data(metadata_path, kmer_path, target_column):
    metadata = pd.read_csv(metadata_path, sep=",", index_col=0)
    kmer_table = pd.read_csv(kmer_path, sep="\t", index_col=0)

    print(f"Loaded metadata shape: {metadata.shape}")
    print(f"Loaded kmer shape: {kmer_table.shape}")

    non_zero_counts = (kmer_table > 0).sum(axis=1)
    cutoff = int(0.05 * kmer_table.shape[1])
    kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]

    print(f"Kmers after filtering: {kmer_filtered.shape[0]}")
    if kmer_filtered.empty:
        raise ValueError("No kmers left after filtering.")

    kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=0), axis=1) * 100
    kmer_normalized = kmer_normalized.astype(float)

    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_normalized.T)
    X = pd.DataFrame(kmer_scaled, index=kmer_normalized.columns)

    shared_samples = metadata.index.intersection(X.index)
    print(f"🔗 Shared samples: {len(shared_samples)}")

    if len(shared_samples) == 0:
        raise ValueError("No shared samples between metadata and kmer data!")

    metadata = metadata.loc[shared_samples]
    X = X.loc[shared_samples]
    y = metadata[target_column]

    return X, y, kmer_normalized, scaler

def compute_balance_metrics(y):
    proportions = y.value_counts(normalize=True)
    gini = 1 - np.sum(proportions ** 2)
    shannon = -np.sum(proportions * np.log(proportions + 1e-12))
    shannon_normalized = shannon / np.log(len(proportions))
    return gini, shannon_normalized, proportions

def choose_best_balancing_method(X, y):
    def apply_and_score(method_name, X_in, y_in):
        if method_name == "oversample":
            df = X_in.copy()
            df["label"] = y_in
            max_count = df["label"].value_counts().max()
            balanced_df = pd.concat([
                resample(group, replace=True, n_samples=max_count, random_state=SEED)
                for _, group in df.groupby("label")
            ])
            X_res = balanced_df.drop(columns=["label"])
            y_res = balanced_df["label"]
        elif method_name == "undersample":
            rus = RandomUnderSampler(random_state=SEED)
            X_res, y_res = rus.fit_resample(X_in, y_in)
        else:
            raise ValueError("Unknown method")

        gini, entropy, _ = compute_balance_metrics(y_res)
        return gini, entropy, X_res, y_res

    methods = ["oversample", "undersample"]
    scores = {}
    results = {}

    print("\n🤜 Trying different class balancing strategies...\n")
    for method in methods:
        gini, entropy, Xb, yb = apply_and_score(method, X, y)
        scores[method] = (gini, entropy)
        results[method] = (Xb, yb)
        print(f"{method.capitalize()} ➔ Gini: {gini:.4f}, Entropy: {entropy:.4f}")

    best_method = max(scores, key=lambda k: (scores[k][0], scores[k][1]))
    print(f"\n✅ Selected balancing method: {best_method.upper()}")
    return results[best_method]

def build_rf():
    return RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED, oob_score=True)

def run_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, None],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    model = GridSearchCV(build_rf(), param_grid, cv=cv, scoring='f1_weighted', verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, labels, name, save_path_prefix):
    print(f"\n{name} Classification Report:")
    report = classification_report(y_true, y_pred, digits=3, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{save_path_prefix}_report.csv")
    print(pd.DataFrame(report).transpose())

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    plt.figure(figsize=(6, 5), dpi=300)
    sns.set_theme(style="white")
    ax = sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', linewidths=0.6, linecolor='gray',
                     xticklabels=labels, yticklabels=labels, square=True, annot_kws={"size": 10, "weight": "bold"})
    ax.set_title(f"{name} Confusion Matrix", fontsize=14, weight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_confusion_matrix.png", dpi=600, bbox_inches='tight')
    plt.close()

def plot_top_features(model, kmer_normalized, path_csv, path_png, top_n=10):
    importances = model.feature_importances_
    feature_names = kmer_normalized.index.tolist()
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_df = importance_df[importance_df["Feature"].str.len() >= 10].sort_values(by="Importance", ascending=False).head(top_n)
    top_df.to_csv(path_csv, index=False)

    plt.figure(figsize=(8, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Importance", y="Feature", data=top_df, palette="Blues_d", edgecolor='black')
    ax.set_title("Top 10 Most Informative k-mers", fontsize=14, weight='bold')
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("k-mer")
    plt.tight_layout()
    plt.savefig(path_png, dpi=600, bbox_inches='tight')
    plt.close()

def compute_and_plot_shap(best_model, X_train, output_path, top_n=10):
    print("\n🌟 Computing SHAP values...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_train)

    plt.figure(figsize=(8, 6), dpi=600)
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=top_n, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ SHAP summary plot saved to {output_path}")

def save_model(model, scaler, prefix, model_dir):
    joblib.dump(model, f"{model_dir}/RF_{prefix}_best_model.joblib")
    joblib.dump(scaler, f"{model_dir}/{prefix}_scaler.joblib")
    pd.DataFrame(model.cv_results_).to_csv(f"{model_dir}/grid_search_results.csv", index=False)

def predict_test(best_model, scaler, kmer_normalized, test_metadata_path, test_kmers_path, target_column):
    metadata = pd.read_csv(test_metadata_path, sep=",", index_col=0)
    kmer = pd.read_csv(test_kmers_path, sep="\t", index_col=0)

    shared_kmers = kmer_normalized.index.intersection(kmer.index)
    kmer = kmer.loc[shared_kmers]
    kmer_norm = kmer.div(kmer.sum(axis=0), axis=1) * 100
    kmer_norm = kmer_norm.astype(float).dropna()
    X_test = pd.DataFrame(scaler.transform(kmer_norm.T), index=kmer.columns)

    shared_samples = metadata.index.intersection(X_test.index)
    metadata = metadata.loc[shared_samples]
    X_test = X_test.loc[shared_samples]
    y_true = metadata[target_column]
    y_pred = best_model.predict(X_test)
    return y_true, y_pred

def plot_hyperparam_performance(grid_model, output_path):
    results = pd.DataFrame(grid_model.cv_results_)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), dpi=600)
    params = ['param_n_estimators', 'param_max_depth', 'param_max_features']
    titles = ['# Estimators', 'Max Depth', 'Max Features']
    best_score = grid_model.best_score_

    for ax, param, title in zip(axs, params, titles):
        x_vals = results[param].astype(str)
        y_vals = results['mean_test_score']
        y_err = results['std_test_score']
        sns.pointplot(x=x_vals, y=y_vals, join=True, ci=None, ax=ax, color='black')
        ax.errorbar(x=np.arange(len(x_vals)), y=y_vals, yerr=y_err, fmt='none', ecolor='gray', capsize=3)
        ax.axhline(best_score, linestyle='--', color='red', label=f'Best F1: {best_score:.4f}')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_title(title)
        ax.set_xlabel(param.replace("param_", "").replace("_", " ").title())
        ax.set_ylabel("Mean F1 Score")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"📈 Hyperparameter performance plot saved to {output_path}")

# === MAIN ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_metadata", required=True)
    parser.add_argument("--test_metadata", required=True)
    parser.add_argument("--train_kmers", required=True)
    parser.add_argument("--test_kmers", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--target_column", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    X, y, kmer_normalized, scaler = load_and_preprocess_data(args.train_metadata, args.train_kmers, args.target_column)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)
    X_train, y_train = choose_best_balancing_method(X_train, y_train)
    grid_model = run_grid_search(X_train, y_train)
    best_model = grid_model.best_estimator_

    save_model(grid_model, scaler, args.target_column.lower(), args.model_dir)
    plot_hyperparam_performance(grid_model, f"{args.output_dir}/hyperparam_performance.png")

    y_pred_val = best_model.predict(X_valid)
    labels = sorted(set(y_valid) | set(y_pred_val))
    evaluate_model(y_valid, y_pred_val, labels, "Validation", f"{args.output_dir}/validation")

    plot_top_features(best_model, kmer_normalized, f"{args.output_dir}/top10_kmers.csv", f"{args.output_dir}/top10_features.png")
    compute_and_plot_shap(best_model, X_train, f"{args.output_dir}/shap_summary.png")

    y_test_true, y_test_pred = predict_test(best_model, scaler, kmer_normalized, args.test_metadata, args.test_kmers, args.target_column)
    labels_test = sorted(set(y_test_true) | set(y_test_pred))
    evaluate_model(y_test_true, y_test_pred, labels_test, "Test", f"{args.output_dir}/test")

    pd.DataFrame({
        f"True_{args.target_column}": y_test_true,
        f"Predicted_{args.target_column}": y_test_pred
    }).to_csv(f"{args.output_dir}/{args.target_column.lower()}_predictions_test.csv")

    print(f"\n✅ {args.target_column} pipeline complete. Outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
