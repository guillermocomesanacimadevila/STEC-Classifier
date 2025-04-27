#!/usr/bin/env python3
# rf_pipeline_with_feature_selection.py

# === IMPORTS ===
import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# === GLOBAL SETTINGS ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
TOP_K_FEATURES = 2000

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

def feature_selection_rf(X, y, top_k=TOP_K_FEATURES, cumulative_threshold=0.95):
    print("\n🚿 Step 1: Applying VarianceThreshold filtering...")
    selector = VarianceThreshold(threshold=0.0001)
    X_reduced = selector.fit_transform(X)
    kept_features = X.columns[selector.get_support()]
    X_reduced = pd.DataFrame(X_reduced, columns=kept_features, index=X.index)
    print(f"✅ Features after VarianceThreshold: {X_reduced.shape[1]}")

    print("\n⚡ Step 2: Removing highly correlated features...")
    corr_matrix = X_reduced.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    X_filtered = X_reduced.drop(columns=to_drop)
    print(f"✅ Features after correlation filtering: {X_filtered.shape[1]}")

    print("\n🌟 Step 3: Training quick RF model for feature importance ranking...")
    rf_temp = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    rf_temp.fit(X_filtered, y)

    importances = rf_temp.feature_importances_
    feature_ranking = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    feature_ranking['Cumulative'] = feature_ranking['Importance'].cumsum()
    feature_ranking.to_csv("full_feature_ranking.csv", index=False)
    print(f"✅ Full feature ranking saved to full_feature_ranking.csv.")

    selected_features = feature_ranking[feature_ranking['Cumulative'] <= cumulative_threshold]['Feature'].tolist()

    if len(selected_features) > top_k:
        selected_features = feature_ranking['Feature'].head(top_k).tolist()
    elif len(selected_features) == 0:
        selected_features = feature_ranking['Feature'].head(top_k).tolist()

    X_selected = X_filtered[selected_features]

    print(f"✅ Selected {len(selected_features)} features (covering {cumulative_threshold*100:.1f}% importance, capped at {top_k}).")

    return X_selected

def compute_balance_metrics(y, title=""):
    proportions = y.value_counts(normalize=True)
    gini = 1 - np.sum(proportions ** 2)
    shannon = -np.sum(proportions * np.log(proportions + 1e-12))
    shannon_normalized = shannon / np.log(len(proportions))
    imbalance_ratio = proportions.max() / proportions.min() if len(proportions) > 1 else 0

    print(f"\n📊 {title} Class Distribution Metrics:")
    print(f"Gini Coefficient: {gini:.4f}")
    print(f"Shannon Entropy (normalized): {shannon_normalized:.4f}")
    print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
    print(f"Class Proportions:\n{proportions}\n")

    return gini, shannon_normalized, imbalance_ratio, proportions

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

        gini, entropy, imbalance, _ = compute_balance_metrics(y_res, title=f"After {method_name.capitalize()}")
        return gini, entropy, imbalance, X_res, y_res

    methods = ["oversample", "undersample"]
    scores = {}
    results = {}

    print("\n🤜 Trying different class balancing strategies...\n")
    for method in methods:
        gini, entropy, imbalance, Xb, yb = apply_and_score(method, X, y)
        scores[method] = (gini, entropy, imbalance)
        results[method] = (Xb, yb)

    best_method = max(scores, key=lambda k: (scores[k][0], scores[k][1], -scores[k][2]))
    print(f"\n✅ Selected balancing method: {best_method.upper()}")
    return results[best_method]

def build_rf():
    return RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED, oob_score=True)

def run_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [10, 20, None],
        'max_features': ['sqrt', 'log2', 0.3, 0.5]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    model = RandomizedSearchCV(
        build_rf(),
        param_distributions=param_grid,
        n_iter=10,
        cv=cv,
        scoring='f1_weighted',
        verbose=1,
        n_jobs=2,
        random_state=SEED
    )
    model.fit(X_train, y_train)

    # 🚨 Optional: auto-save grid_model immediately after fit, in case crash happens
    joblib.dump(model, "temporary_grid_model.pkl")
    print("✅ Temporary grid search object saved after fitting.")

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

def save_model(grid_model, scaler, prefix, model_dir):
    best_model = grid_model.best_estimator_

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_model, f"{model_dir}/RF_{prefix}_best_model.pkl")
    joblib.dump(scaler, f"{model_dir}/{prefix}_scaler.pkl")

    pd.DataFrame(grid_model.cv_results_).to_csv(f"{model_dir}/grid_search_results.csv", index=False)

    with open(f"{model_dir}/model_summary.txt", 'w') as f:
        f.write(f"Best Hyperparameters:\n{grid_model.best_params_}\n\n")
        f.write(f"OOB Score: {getattr(best_model, 'oob_score_', 'N/A')}\n")

    joblib.dump(grid_model, f"{model_dir}/full_grid_model.pkl")

def predict_test(best_model, scaler, kmer_normalized, test_metadata_path, test_kmers_path, target_column, selected_features):
    metadata = pd.read_csv(test_metadata_path, sep=",", index_col=0)
    kmer = pd.read_csv(test_kmers_path, sep="\t", index_col=0)

    shared_kmers = kmer_normalized.index.intersection(kmer.index)
    kmer = kmer.loc[shared_kmers]
    kmer_norm = kmer.div(kmer.sum(axis=0), axis=1) * 100
    kmer_norm = kmer_norm.astype(float).dropna()

    X_test = pd.DataFrame(scaler.transform(kmer_norm.T), index=kmer.columns)
    X_test = X_test[selected_features]

    shared_samples = metadata.index.intersection(X_test.index)
    metadata = metadata.loc[shared_samples]
    X_test = X_test.loc[shared_samples]
    y_true = metadata[target_column]
    y_pred = best_model.predict(X_test)
    return y_true, y_pred

def plot_top_features(model, feature_names, path_csv, path_png, top_n=10):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    top_df.to_csv(path_csv, index=False)

    plt.figure(figsize=(8, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Importance", y="Feature", data=top_df, edgecolor='black')
    ax.set_title("Top 10 Most Informative Features", fontsize=14, weight='bold')
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(path_png, dpi=600, bbox_inches='tight')
    plt.close()

def compute_and_plot_rf_importance(best_model, X_train, output_path, top_n=10):
    print("\n🌟 Plotting RF feature importances...")
    importances = best_model.feature_importances_
    plt.figure(figsize=(8, 6), dpi=600)
    indices = np.argsort(importances)[-top_n:]
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(np.arange(top_n), np.array(X_train.columns)[indices])
    plt.xlabel('Relative Importance')
    plt.title('Top RF Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ RF feature importance plot saved to {output_path}")

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_metadata", required=True)
    parser.add_argument("--test_metadata", required=True)
    parser.add_argument("--train_kmers", required=True)
    parser.add_argument("--test_kmers", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    for target_column in ["Region", "Country"]:
        output_dir = os.path.join(args.output_dir, target_column.lower())
        model_dir = os.path.join(args.model_dir, target_column.lower())

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # === Load and preprocess data ===
        X, y, kmer_normalized, scaler = load_and_preprocess_data(args.train_metadata, args.train_kmers, target_column)
        compute_balance_metrics(y, title=f"{target_column} Before Balancing")

        # === Feature selection ===
        X_selected = feature_selection_rf(X, y, top_k=TOP_K_FEATURES)

        # === Class balancing ===
        X_balanced, y_balanced = choose_best_balancing_method(X_selected, y)

        # === Train/validation split ===
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_balanced, y_balanced,
            stratify=y_balanced,
            test_size=0.2,
            random_state=SEED
        )

        # === Hyperparameter tuning ===
        grid_model = run_grid_search(X_train, y_train)
        best_model = grid_model.best_estimator_

        # === Save model, scaler, and grid search results ===
        save_model(grid_model, scaler, target_column.lower(), model_dir)

        # (❌ REMOVED hyperparameter performance plot)

        # === Validation prediction and evaluation ===
        y_pred_val = best_model.predict(X_valid)
        labels_val = sorted(set(y_valid) | set(y_pred_val))
        evaluate_model(y_valid, y_pred_val, labels_val, "Validation", os.path.join(output_dir, "validation"))

        # === Feature importance plots ===
        plot_top_features(best_model, X_selected.columns, os.path.join(output_dir, "top10_kmers.csv"), os.path.join(output_dir, "top10_features.png"))
        compute_and_plot_rf_importance(best_model, X_train, os.path.join(output_dir, "rf_summary.png"))

        # === Test set prediction and evaluation ===
        y_test_true, y_test_pred = predict_test(
            best_model,
            scaler,
            kmer_normalized,
            args.test_metadata,
            args.test_kmers,
            target_column,
            selected_features=X_selected.columns
        )
        labels_test = sorted(set(y_test_true) | set(y_test_pred))
        evaluate_model(y_test_true, y_test_pred, labels_test, "Test", os.path.join(output_dir, "test"))

        # === Save predictions on test set ===
        pd.DataFrame({
            f"True_{target_column}": y_test_true,
            f"Predicted_{target_column}": y_test_pred
        }).to_csv(os.path.join(output_dir, f"{target_column.lower()}_predictions_test.csv"))

        print(f"\n✅ {target_column} pipeline complete. Outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()
