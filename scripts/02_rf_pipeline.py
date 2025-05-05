#!/usr/bin/env python3

# === IMPORTS ===
import os
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# === GLOBAL SETTINGS ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
MAX_FEATURES = 1000
CUMULATIVE_THRESHOLD = 0.98

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

    kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=1), axis=0) * 100
    kmer_normalized = kmer_normalized.astype(float)

    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_normalized)
    X = pd.DataFrame(kmer_scaled, index=kmer_normalized.index, columns=kmer_normalized.columns)

    shared_samples = metadata.index.intersection(X.index)
    print(f"🔗 Shared samples: {len(shared_samples)}")

    if len(shared_samples) == 0:
        raise ValueError("No shared samples between metadata and kmer data!")

    metadata = metadata.loc[shared_samples]
    X = X.loc[shared_samples]
    y = metadata[target_column]

    return X, y, kmer_normalized, scaler, kmer_filtered

def feature_selection_rf(X, y, kmer_filtered):
    print("\n🚿 Applying VarianceThreshold filtering...")
    selector = VarianceThreshold(threshold=0.0001)
    X_reduced = selector.fit_transform(X)
    kept_features = X.columns[selector.get_support()]
    X_reduced = pd.DataFrame(X_reduced, columns=kept_features, index=X.index)
    print(f"✅ Features after VarianceThreshold: {X_reduced.shape[1]}")

    print("\n📉 Removing highly correlated features...")
    X_np = X_reduced.to_numpy()
    X_np -= np.mean(X_np, axis=0)
    X_np /= np.std(X_np, axis=0)

    cor_matrix = np.abs(np.corrcoef(X_np, rowvar=False))
    upper_tri = np.triu(cor_matrix, k=1)
    to_drop = [X_reduced.columns[j] for i in range(upper_tri.shape[0]) for j in range(i + 1, upper_tri.shape[1]) if upper_tri[i, j] > 0.90]
    X_filtered = X_reduced.drop(columns=to_drop)
    print(f"✅ Features after correlation filtering: {X_filtered.shape[1]}")

    print("\n🌲 Training RF for feature importance ranking...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=SEED, n_jobs=-1, class_weight='balanced_subsample')
    rf.fit(X_filtered, y)

    importances = rf.feature_importances_
    feature_ranking = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    feature_ranking['Cumulative'] = feature_ranking['Importance'].cumsum()
    feature_ranking['Kmer_sequence'] = feature_ranking['Feature']
    feature_ranking.to_csv("full_feature_ranking.csv", index=False)
    print("✅ Saved full feature ranking to full_feature_ranking.csv")

    selected = feature_ranking[feature_ranking['Cumulative'] <= CUMULATIVE_THRESHOLD].head(MAX_FEATURES)
    X_selected = X_filtered[selected['Feature']]

    print(f"✅ Selected {X_selected.shape[1]} features.")
    return X_selected, feature_ranking

def compute_balance_metrics(y, title="", save_path=None):
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

    if save_path:
        metrics_df = pd.DataFrame({
            "Metric": ["Gini", "Shannon Normalized", "Imbalance Ratio"],
            "Value": [gini, shannon_normalized, imbalance_ratio]
        })
        metrics_df.to_csv(save_path, index=False)
        print(f"✅ Balance metrics saved to {save_path}")

    return gini, shannon_normalized, imbalance_ratio, proportions

def apply_smote(X, y):
    print("\n⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def run_grid_search(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'max_features': ['sqrt', 0.3],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    model = RandomizedSearchCV(
        RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=SEED, oob_score=True),
        param_distributions=param_grid,
        n_iter=10,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring='f1_weighted',
        verbose=1,
        n_jobs=-1,
        random_state=SEED
    )
    model.fit(X_train, y_train)
    joblib.dump(model, "temporary_grid_model.pkl")
    print("✅ Grid search complete and saved.")
    return model

def evaluate_model(y_true, y_pred, labels, name, save_prefix):
    print(f"\n🧪 {name} Classification Report:")

    # Classification report
    report = classification_report(y_true, y_pred, digits=3, output_dict=True, labels=labels)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{save_prefix}_report.csv")
    print(report_df)

    # Accuracy (explicit)
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n✅ Accuracy: {accuracy:.4f}")

    # Confusion matrix (raw)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_df.to_csv(f"{save_prefix}_confusion_matrix_raw.csv")

    # Confusion matrix (normalized row-wise)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
    conf_df_norm = pd.DataFrame(np.round(conf_matrix_norm, 3), index=labels, columns=labels)
    conf_df_norm.to_csv(f"{save_prefix}_confusion_matrix_normalized.csv")

    # Plot: raw confusion matrix
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=600)
    plt.close()

    # Plot: normalized confusion matrix
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(conf_df_norm, annot=True, fmt='.2f', cmap='Purples', xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix_normalized.png", dpi=600)
    plt.close()

    print(f"✅ All metrics saved for {name} to:\n  → {save_prefix}_report.csv\n  → {save_prefix}_confusion_matrix_[...].csv/png")

def plot_top_features(model, feature_names, feature_ranking, path_csv, path_png, top_n=10):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    top_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)
    top_df = top_df.merge(feature_ranking[['Feature', 'Kmer_sequence']], on="Feature", how="left")
    top_df.to_csv(path_csv, index=False)

    plt.figure(figsize=(8, 5), dpi=300)
    sns.barplot(x="Importance", y="Kmer_sequence", data=top_df, edgecolor='black')
    plt.title("Top 10 Most Informative K-mers")
    plt.tight_layout()
    plt.savefig(path_png, dpi=600)
    plt.close()
    print(f"✅ Saved top feature plot: {path_png}")

def compute_and_plot_rf_importance(best_model, X_train, output_path, top_n=10):
    print("\n🌟 Plotting RF feature importances...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6), dpi=600)
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(np.arange(top_n), np.array(X_train.columns)[indices])
    plt.xlabel('Relative Importance')
    plt.title('Top RF Feature Importances')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ RF feature importance plot saved to {output_path}")

def save_model(grid_model, scaler, prefix, model_dir, selected_features):
    best_model = grid_model.best_estimator_

    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/RF_{prefix}_best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ Best RF model saved to {model_path}")

    scaler_path = f"{model_dir}/{prefix}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved to {scaler_path}")

    features_path = f"{model_dir}/{prefix}_selected_features.pkl"
    joblib.dump(selected_features, features_path)
    print(f"✅ Selected feature list saved to {features_path}")

    grid_search_csv_path = f"{model_dir}/grid_search_results.csv"
    pd.DataFrame(grid_model.cv_results_).to_csv(grid_search_csv_path, index=False)
    print(f"✅ Grid search CV results saved to {grid_search_csv_path}")

    oob_score = getattr(best_model, 'oob_score_', 'N/A')
    summary_path = f"{model_dir}/model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Best Hyperparameters:\n{grid_model.best_params_}\n\n")
        f.write(f"OOB Score: {oob_score}\n")
    print(f"✅ Model summary saved to {summary_path}")

def predict_test(best_model, scaler, kmer_normalized, test_metadata_path, test_kmers_path, target_column, selected_features):
    metadata = pd.read_csv(test_metadata_path, sep=",", index_col=0)
    kmer = pd.read_csv(test_kmers_path, sep="\\t", index_col=0)

    shared_kmers = kmer_normalized.index.intersection(kmer.index)
    kmer = kmer.loc[shared_kmers]
    kmer_norm = kmer.div(kmer.sum(axis=1), axis=0) * 100
    kmer_norm = kmer_norm.astype(float).dropna()

    X_test = pd.DataFrame(scaler.transform(kmer_norm), index=kmer_norm.index, columns=kmer_norm.columns)
    for feat in selected_features:
        if feat not in X_test.columns:
            X_test[feat] = 0
    X_test = X_test[selected_features]

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
    parser = argparse.ArgumentParser(description="Final Random Forest pipeline for classification using k-mers and metadata.")
    parser.add_argument("--train_metadata", required=True)
    parser.add_argument("--test_metadata", required=True)
    parser.add_argument("--train_kmers", required=True)
    parser.add_argument("--test_kmers", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    for target_column in ["Region", "Country"]:
        print(f"\n=== 🔁 Pipeline for: {target_column.upper()} ===")
        output_dir = os.path.join(args.output_dir, target_column.lower())
        model_dir = os.path.join(args.model_dir, target_column.lower())
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        t0 = time.time()
        X, y, kmer_normalized, scaler, kmer_filtered = load_and_preprocess_data(args.train_metadata, args.train_kmers, target_column)
        t1 = time.time()
        print(f"⏱️ Preprocessing time: {t1 - t0:.2f} seconds")

        compute_balance_metrics(y, title="Before Balancing", save_path=f"{output_dir}/balance_before.csv")

        t2 = time.time()
        X_selected, feature_ranking = feature_selection_rf(X, y, kmer_filtered)
        t3 = time.time()
        print(f"⏱️ Feature selection time: {t3 - t2:.2f} seconds")

        compute_balance_metrics(y, title="Before Balancing (selected)", save_path=f"{output_dir}/balance_selected.csv")

        X_balanced, y_balanced = apply_smote(X_selected, y)
        compute_balance_metrics(y_balanced, title="After SMOTE", save_path=f"{output_dir}/balance_after.csv")

        X_train, X_valid, y_train, y_valid = train_test_split(X_balanced, y_balanced, stratify=y_balanced, test_size=0.2, random_state=SEED)
        print("\n✅ Validation set class distribution:")
        print(y_valid.value_counts(normalize=True))

        t4 = time.time()
        grid_model = run_grid_search(X_train, y_train)
        t5 = time.time()
        print(f"⏱️ Grid search time: {t5 - t4:.2f} seconds")

        best_model = grid_model.best_estimator_

        save_model(grid_model, scaler, target_column.lower(), model_dir, X_selected.columns.tolist())
        plot_hyperparam_performance(grid_model, f"{output_dir}/hyperparam_performance.png")

        y_pred_val = best_model.predict(X_valid)
        labels_val = sorted(set(y_valid) | set(y_pred_val))
        evaluate_model(y_valid, y_pred_val, labels_val, "Validation", f"{output_dir}/validation")

        plot_top_features(best_model, X_selected.columns, feature_ranking, f"{output_dir}/top10_kmers.csv", f"{output_dir}/top10_features.png")
        compute_and_plot_rf_importance(best_model, X_train, f"{output_dir}/rf_summary.png")

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
        evaluate_model(y_test_true, y_test_pred, labels_test, "Test", f"{output_dir}/test")

        pd.DataFrame({
            f"True_{target_column}": y_test_true,
            f"Predicted_{target_column}": y_test_pred
        }).to_csv(f"{output_dir}/{target_column.lower()}_predictions_test.csv")

        final_cv_scores = cross_val_score(
            best_model,
            X_balanced,
            y_balanced,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring='f1_macro',
            n_jobs=-1
        )
        print(f"\n✅ Final 5-Fold F1_macro ({target_column}): {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")
        print(f"✅ {target_column} pipeline complete. Output saved to {output_dir}\\n")

if __name__ == "__main__":
    main()
