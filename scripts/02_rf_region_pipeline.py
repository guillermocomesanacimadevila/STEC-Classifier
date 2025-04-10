#!/usr/bin/env python3

import os
import warnings
import pickle
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === CLI ARGUMENTS ===
parser = argparse.ArgumentParser(description="Train RF model and evaluate on test data")
parser.add_argument("--train_metadata", required=True)
parser.add_argument("--train_kmers", required=True)
parser.add_argument("--test_metadata", required=True)
parser.add_argument("--test_kmers", required=True)
parser.add_argument("--output_dir", default="output")
parser.add_argument("--model_dir", default="models")
parser.add_argument("--target_column", default="Region")
args = parser.parse_args()

# === CONFIG ===
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)

# === Load Data ===
metadata = pd.read_csv(args.train_metadata, sep=",", index_col=0)
kmer_table = pd.read_csv(args.train_kmers, sep="\t", index_col=0)

# === Filter kmers (remove rare ones) ===
non_zero_counts = (kmer_table > 0).sum(axis=1)
cutoff = int(0.05 * kmer_table.shape[1])
kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]
if kmer_filtered.empty:
    raise ValueError("❌ No kmers left after filtering.")

# === Normalize ===
kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=0), axis=1) * 100
kmer_normalized = kmer_normalized.astype(float)

scaler = StandardScaler()
kmer_scaled = scaler.fit_transform(kmer_normalized.T)
X = pd.DataFrame(kmer_scaled, index=kmer_normalized.columns)

# === Match samples between metadata and k-mers ===
shared_samples = metadata.index.intersection(X.index)

print(f"Samples in metadata: {len(metadata.index)}")
print(f"Samples in kmers: {len(X.index)}")
print(f"Shared samples: {len(shared_samples)}")

if len(shared_samples) == 0:
    raise ValueError("❌ No shared samples between metadata and kmer data!")

metadata = metadata.loc[shared_samples]
X = X.loc[shared_samples]
y = metadata[args.target_column]

# === Train/Test Split ===
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Model Training ===
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 30],
    'max_features': ['sqrt', 'log2']
}
model = GridSearchCV(
    RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1),
    param_grid, cv=3, scoring='accuracy', verbose=1
)
model.fit(X_train, y_train)
best_model = model.best_estimator_

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
print(f"\n🔍 Cross-validation Accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# === Save Model and Scaler ===
joblib.dump(best_model, f"{args.model_dir}/RF_{args.target_column.lower()}_best_model.joblib")
joblib.dump(scaler, f"{args.model_dir}/{args.target_column.lower()}_scaler.joblib")

with open(f"{args.model_dir}/rf_{args.target_column.lower()}_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{args.model_dir}/{args.target_column.lower()}_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === Evaluate on Validation ===
y_pred = best_model.predict(X_valid)
print("\n📊 Classification Report (Validation):")
print(classification_report(y_valid, y_pred))
print(f"F1 Score: {f1_score(y_valid, y_pred, average='weighted'):.3f}")
print(f"Precision: {precision_score(y_valid, y_pred, average='weighted'):.3f}")
print(f"Recall: {recall_score(y_valid, y_pred, average='weighted'):.3f}")

# === Confusion Matrix (Validation) ===
labels = sorted(set(y_valid) | set(y_pred))
conf_matrix = confusion_matrix(y_valid, y_pred, labels=labels)
conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
plt.title("Validation Confusion Matrix")
plt.ylabel(f"True {args.target_column}")
plt.xlabel(f"Predicted {args.target_column}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/confusion_matrix_validation.png")
plt.close()

# === Feature Importances
importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'features': kmer_normalized.index,
    'importance': importances
}).sort_values(by='importance', ascending=False)

top10 = importance_df[importance_df['features'].str.len() >= 10].head(10)
top10.to_csv(f"{args.output_dir}/top10_kmers.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(range(len(top10)), top10['importance'], tick_label=top10['features'])
plt.xticks(rotation=90)
plt.ylabel("Mean Decrease in Accuracy")
plt.title("Top 10 Features for Classification")
plt.tight_layout()
plt.savefig(f"{args.output_dir}/top10_features.png")
plt.close()

# === Predict on Test Data
metadata19 = pd.read_csv(args.test_metadata, sep=",", index_col=0)
kmer19 = pd.read_csv(args.test_kmers, sep="\t", index_col=0)

shared_kmers = kmer_normalized.index.intersection(kmer19.index)
if shared_kmers.empty:
    raise ValueError("❌ No shared kmers between training and test data!")

kmer19 = kmer19.loc[shared_kmers]
kmer19_norm = kmer19.div(kmer19.sum(axis=0), axis=1) * 100
kmer19_norm = kmer19_norm.astype(float).dropna()
X_2019 = pd.DataFrame(scaler.transform(kmer19_norm.T), index=kmer19.columns)

shared_samples_2019 = metadata19.index.intersection(X_2019.index)

if len(shared_samples_2019) == 0:
    raise ValueError("❌ No shared samples between test metadata and kmer data!")

metadata19 = metadata19.loc[shared_samples_2019]
X_2019 = X_2019.loc[shared_samples_2019]
y_2019_true = metadata19[args.target_column]
y_2019_pred = best_model.predict(X_2019)

pred_df = pd.DataFrame({
    f"True_{args.target_column}": y_2019_true,
    f"Predicted_{args.target_column}": y_2019_pred
})
pred_df.to_csv(f"{args.output_dir}/{args.target_column.lower()}_predictions_2019.csv")

# === Confusion Matrix (Test Set)
labels_2019 = sorted(set(y_2019_true) | set(y_2019_pred))
conf_matrix_2019 = confusion_matrix(y_2019_true, y_2019_pred, labels=labels_2019)
conf_df_2019 = pd.DataFrame(conf_matrix_2019, index=labels_2019, columns=labels_2019)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df_2019, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f"2019 Confusion Matrix: {args.target_column}")
plt.ylabel(f"True {args.target_column}")
plt.xlabel(f"Predicted {args.target_column}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{args.output_dir}/confusion_matrix_2019.png")
plt.close()

print("\n✅ Model training and evaluation complete. Outputs saved.")