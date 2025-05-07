# === IMPORTS ===
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


warnings.filterwarnings("ignore")

# === GLOBAL SETTINGS ===
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
MAX_FEATURES = 1000
CUMULATIVE_THRESHOLD = 0.98

def load_and_preprocess_data(metadata_path, kmer_path, target_column):
    metadata = pd.read_csv(metadata_path, sep=",", index_col=0)
    kmer_table = pd.read_csv(kmer_path, sep="\t", index_col=0)

    print(f"Loaded metadata shape: {metadata.shape}")
    print(f"Loaded kmer shape: {kmer_table.shape}")

    # Print example index entries for debugging
    print(f"⚠️   First few metadata sample IDs: {metadata.index[:5].tolist()}")
    print(f"⚠️   First few kmer sample IDs: {kmer_table.index[:5].tolist()}")

    # Transpose k-mer table if index appears to be kmers instead of sample names
    if not kmer_table.index.str.startswith('srr').any():
        print("🔄 Transposing kmer table (index doesn't look like sample IDs)...")
        kmer_table = kmer_table.transpose()

    # Normalize sample IDs
    metadata.index = metadata.index.str.strip().str.lower()
    kmer_table.index = kmer_table.index.str.strip().str.lower()

    # Find shared sample IDs
    shared_samples = metadata.index.intersection(kmer_table.index)
    print(f"Shared samples: {len(shared_samples)}")

    if len(shared_samples) == 0:
        raise ValueError("No shared samples between metadata and kmer data!")

    # Filter both tables to shared sample IDs
    metadata = metadata.loc[shared_samples]
    kmer_table = kmer_table.loc[shared_samples]

    # Filter out sparse rows
    non_zero_counts = (kmer_table > 0).sum(axis=1)
    cutoff = int(0.05 * kmer_table.shape[1])
    kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]
    print(f"Kmers after filtering: {kmer_filtered.shape[0]}")
    if kmer_filtered.empty:
        raise ValueError("No kmers left after filtering.")

    # Normalize row-wise
    kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=1), axis=0) * 100
    kmer_normalized = kmer_normalized.astype(float)

    # Standard scale
    scaler = StandardScaler()
    kmer_scaled = scaler.fit_transform(kmer_normalized)
    X = pd.DataFrame(kmer_scaled, index=kmer_normalized.index, columns=kmer_normalized.columns)

    # Get labels
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
    to_drop = [X_reduced.columns[j]
               for i in range(upper_tri.shape[0])
               for j in range(i + 1, upper_tri.shape[1])
               if upper_tri[i, j] > 0.90]
    X_filtered = X_reduced.drop(columns=to_drop)
    print(f"✅ Features after correlation filtering: {X_filtered.shape[1]}")

    # Align y with X_filtered (ensure sample counts match)
    y_aligned = y.loc[X_filtered.index]

    print("\n🌲 Training RF for feature importance ranking...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    rf.fit(X_filtered, y_aligned)

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

    # Ensure final output has aligned y as well
    y_selected = y_aligned.loc[X_selected.index]

    print(f"✅ Selected {X_selected.shape[1]} features.")
    return X_selected, y_selected, feature_ranking


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


def apply_best_balancing(X, y):
    print("\n⚖️ Testing balancing strategies (SMOTE, Over, Under)...")
    strategies = {
        'SMOTE': SMOTE(random_state=SEED),
        'OverSampling': RandomOverSampler(random_state=SEED),
        'UnderSampling': RandomUnderSampler(random_state=SEED)
    }
    best_score = -np.inf
    best_X, best_y = X, y
    best_name = None

    for name, sampler in strategies.items():
        try:
            X_res, y_res = sampler.fit_resample(X, y)
            proportions = y_res.value_counts(normalize=True)
            gini = 1 - np.sum(proportions ** 2)
            shannon = -np.sum(proportions * np.log(proportions + 1e-12)) / np.log(len(proportions))
            score = gini + shannon
            print(f"{name} - Gini: {gini:.4f}, Shannon Norm: {shannon:.4f}, Composite: {score:.4f}")

            if score > best_score:
                best_score = score
                best_X, best_y = X_res, y_res
                best_name = name
        except Exception as e:
            print(f"⚠️ {name} failed: {e}")

    print(f"\n✅ Best balancing method: {best_name} with composite score: {best_score:.4f}")
    return best_X, best_y

# === REPLACE apply_smote ===
def apply_smote(X, y):
    return apply_best_balancing(X, y)

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

def find_best_split(X, y, seed=42):
    print("\n🔍 Finding best train-test split based on F1-weighted score...")
    best_score = 0
    best_split = None
    best_data = {}

    for test_size in [0.3, 0.25, 0.2]:
        print(f"\n⏳ Testing split: {int((1 - test_size) * 100)}:{int(test_size * 100)}")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=seed
        )

        X_train_smote, y_train_smote = apply_smote(X_train, y_train)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=seed,
            n_jobs=-1
        )
        model.fit(X_train_smote, y_train_smote)
        y_pred = model.predict(X_valid)

        score = f1_score(y_valid, y_pred, average='weighted')
        print(f"✅ F1-weighted: {score:.4f}")

        if score > best_score:
            best_score = score
            best_split = test_size
            best_data = {
                'X_train': X_train_smote,
                'y_train': y_train_smote,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'model': model
            }

    print(f"\n🏆 Best split: {int((1 - best_split) * 100)}:{int(best_split * 100)} with F1-weighted score = {best_score:.4f}")
    return best_split, best_data

def evaluate_model(y_true, y_pred, labels, name, save_prefix):
    print(f"\n🧪 {name} Classification Report:")

    report = classification_report(y_true, y_pred, digits=3, output_dict=True, labels=labels)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{save_prefix}_report.csv")
    print(report_df)

    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n✅ Accuracy: {accuracy:.4f}")

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_df.to_csv(f"{save_prefix}_confusion_matrix_raw.csv")

    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)
    conf_df_norm = pd.DataFrame(np.round(conf_matrix_norm, 3), index=labels, columns=labels)
    conf_df_norm.to_csv(f"{save_prefix}_confusion_matrix_normalized.csv")

    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png", dpi=600)
    plt.close()

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

def predict_test(best_model, scaler, kmer_normalized_train, test_metadata_path, test_kmers_path, target_column, selected_features):
    import pandas as pd

    # Load test metadata and k-mer table
    metadata = pd.read_csv(test_metadata_path, sep=",", index_col=0)
    kmer = pd.read_csv(test_kmers_path, sep="\t", index_col=0)

    # Normalize sample IDs
    metadata.index = metadata.index.str.strip().str.lower()
    print(f"⚠️     First test metadata IDs: {metadata.index.tolist()[:5]}")
    print(f"⚠️     First test k-mer IDs: {kmer.index.tolist()[:5]}")
    print(f"⚠️     Training k-mer IDs: {kmer_normalized_train.index.tolist()[:5]}")

    # Transpose k-mer table if index doesn't look like sample IDs
    if not kmer.index.str.startswith('srr').any():
        print("🔄 Transposing test k-mer table (index doesn't look like sample IDs)...")
        kmer = kmer.T

    # Normalize transposed index
    kmer.index = kmer.index.str.strip().str.lower()

    # Match shared samples
    shared_samples = metadata.index.intersection(kmer.index)
    print(f"✅ Shared samples between test metadata and k-mer: {len(shared_samples)}")

    if len(shared_samples) == 0:
        raise ValueError("No matching test samples found between metadata and kmer data!")

    # Filter metadata and kmers to shared samples
    metadata = metadata.loc[shared_samples]
    kmer = kmer.loc[shared_samples]

    # Normalize rows
    kmer_norm = kmer.div(kmer.sum(axis=1), axis=0) * 100
    kmer_norm = kmer_norm.astype(float)

    # Add missing training features to test set
    for feat in scaler.feature_names_in_:
        if feat not in kmer_norm.columns:
            kmer_norm[feat] = 0.0
    kmer_norm = kmer_norm[scaler.feature_names_in_]

    # Apply standard scaling
    X_test_scaled = pd.DataFrame(
        scaler.transform(kmer_norm),
        index=kmer_norm.index,
        columns=scaler.feature_names_in_
    )

    # Keep only selected features
    X_test = X_test_scaled[selected_features]

    # Predict
    y_true = metadata[target_column]
    y_pred = best_model.predict(X_test)

    return y_true, y_pred


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

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Random Forest pipeline on k-mer data.")
    parser.add_argument("--train_metadata", required=True, help="Path to training metadata CSV file")
    parser.add_argument("--train_kmers", required=True, help="Path to training k-mer table TSV file")
    parser.add_argument("--test_metadata", required=True, help="Path to test metadata CSV file")
    parser.add_argument("--test_kmers", required=True, help="Path to test k-mer table TSV file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save output")
    parser.add_argument("--model_dir", default="./models", help="Directory to save models")
    parser.add_argument("--targets", nargs="+", default=["Region", "Country"], help="List of target column names")
    args = parser.parse_args()

    for target_column in args.targets:
        print(f"\n=== 🔁 Pipeline for: {target_column.upper()} ===")
        output_dir_target = os.path.join(args.output_dir, target_column.lower())
        model_dir_target = os.path.join(args.model_dir, target_column.lower())

        os.makedirs(output_dir_target, exist_ok=True)
        os.makedirs(model_dir_target, exist_ok=True)

        t0 = time.time()
        X, y, kmer_normalized, scaler, kmer_filtered = load_and_preprocess_data(
            args.train_metadata, args.train_kmers, target_column
        )
        t1 = time.time()
        print(f"⏱️  Preprocessing time: {t1 - t0:.2f} seconds")

        compute_balance_metrics(y, title="Before Balancing", save_path=f"{output_dir_target}/balance_before.csv")

        t2 = time.time()
        X_selected, y_selected, feature_ranking = feature_selection_rf(X, y, kmer_filtered)
        t3 = time.time()
        print(f"⏱️  Feature selection time: {t3 - t2:.2f} seconds")

        compute_balance_metrics(y, title="Before Balancing (selected)", save_path=f"{output_dir_target}/balance_selected.csv")

        best_split, best_data = find_best_split(X_selected, y_selected)

        X_train = best_data['X_train']
        y_train = best_data['y_train']
        X_valid = best_data['X_valid']
        y_valid = best_data['y_valid']

        compute_balance_metrics(y_train, title="After Balancing", save_path=f"{output_dir_target}/balance_after.csv")

        print("\n✅ Validation set class distribution:")
        print(y_valid.value_counts(normalize=True))

        t4 = time.time()
        grid_model = run_grid_search(X_train, y_train)
        t5 = time.time()
        print(f"⏱️  Grid search time: {t5 - t4:.2f} seconds")

        best_model = grid_model.best_estimator_

        save_model(grid_model, scaler, target_column.lower(), model_dir_target, X_selected.columns.tolist())

        plot_hyperparam_performance(grid_model, f"{output_dir_target}/hyperparam_performance.png")

        y_pred_val = best_model.predict(X_valid)
        labels_val = sorted(set(y_valid) | set(y_pred_val))
        evaluate_model(y_valid, y_pred_val, labels_val, "Validation", f"{output_dir_target}/validation")

        plot_top_features(
            best_model,
            X_selected.columns,
            feature_ranking,
            f"{output_dir_target}/top10_kmers.csv",
            f"{output_dir_target}/top10_features.png"
        )

        compute_and_plot_rf_importance(best_model, X_train, f"{output_dir_target}/rf_summary.png")

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
        evaluate_model(y_test_true, y_test_pred, labels_test, "Test", f"{output_dir_target}/test")

        pd.DataFrame({
            f"True_{target_column}": y_test_true,
            f"Predicted_{target_column}": y_test_pred
        }).to_csv(f"{output_dir_target}/{target_column.lower()}_predictions_test.csv")

        final_cv_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring='f1_macro',
            n_jobs=-1
        )

        print(f"\n✅ Final 5-Fold F1_macro ({target_column}): {final_cv_scores.mean():.3f} ± {final_cv_scores.std():.3f}")
        print(f"✅ {target_column} pipeline complete. Output saved to {output_dir_target}\n")

if __name__ == "__main__":
    main()
