#!/usr/bin/env python3

import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# === Load and preprocess 2014–2018 training data ===
metadata = pd.read_csv("data/Training/14-18metadata", sep=",", index_col=0)
kmer_table = pd.read_csv("data/Training/14-18kmerdata.txt", sep="\t", index_col=0)

non_zero_counts = (kmer_table > 0).sum(axis=1)
cutoff = int(0.05 * kmer_table.shape[1])
kmer_filtered = kmer_table.loc[non_zero_counts > cutoff]
if kmer_filtered.empty:
    raise ValueError("❌ No kmers left after filtering.")

kmer_normalized = kmer_filtered.div(kmer_filtered.sum(axis=0), axis=1) * 100
kmer_normalized = kmer_normalized.astype(float)

scaler = StandardScaler()
kmer_scaled = scaler.fit_transform(kmer_normalized.T)
X = pd.DataFrame(kmer_scaled, index=kmer_normalized.columns)
y = metadata.loc[X.index, "Region"]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 30],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42, oob_score=True, n_jobs=-1),
                           param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Save model and scaler
joblib.dump(best_model, "models/RF_region_best_model.joblib")
joblib.dump(scaler, "models/region_scaler.joblib")
with open("models/rf_region_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/region_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# === Evaluate on validation ===
y_pred = best_model.predict(X_valid)
print("\n📊 Classification Report (Validation):\n")
print(classification_report(y_valid, y_pred))

labels = sorted(y.unique())
conf_matrix = confusion_matrix(y_valid, y_pred, labels=labels)
conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
plt.title("Validation Confusion Matrix")
plt.ylabel("True Region")
plt.xlabel("Predicted Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/confusion_matrix_validation.png")
plt.close()

# === Feature importance ===
importances = best_model.feature_importances_
importance_df = pd.DataFrame({
    'features': kmer_normalized.index,
    'importance': importances
}).sort_values(by='importance', ascending=False)

# Filter valid kmers (length > 10 bp) and save top 10
top10 = importance_df[importance_df['features'].str.len() >= 10].head(10)
top10.to_csv("output/top10_kmers.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(range(len(top10)), top10['importance'], tick_label=top10['features'])
plt.xticks(rotation=90)
plt.ylabel("Mean Decrease in Accuracy")
plt.title("Top 10 Features for Region Classification")
plt.tight_layout()
plt.savefig("output/top10_features.png")
plt.close()

# === Predict on 2019 test data ===
metadata19 = pd.read_csv("data/Testing/19metadata", sep=",", index_col=0)
kmer19 = pd.read_csv("data/Testing/19kmerdata.txt", sep="\t", index_col=0)

shared_kmers = kmer_normalized.index.intersection(kmer19.index)
if shared_kmers.empty:
    raise ValueError("❌ No shared kmers between training and 2019 data!")

kmer19 = kmer19.loc[shared_kmers]
kmer19_norm = kmer19.div(kmer19.sum(axis=0), axis=1) * 100
kmer19_norm = kmer19_norm.astype(float).dropna()
X_2019 = pd.DataFrame(scaler.transform(kmer19_norm.T), index=kmer19.columns)

y_2019_true = metadata19.loc[X_2019.index, "Region"]
y_2019_pred = best_model.predict(X_2019)

pred_df = pd.DataFrame({
    "True_Region": y_2019_true,
    "Predicted_Region": y_2019_pred
})
pred_df.to_csv("output/region_predictions_2019.csv")

# === 2019 Confusion Matrix ===
labels_2019 = sorted(y_2019_true.unique())
conf_matrix_2019 = confusion_matrix(y_2019_true, y_2019_pred, labels=labels_2019)
conf_df_2019 = pd.DataFrame(conf_matrix_2019, index=labels_2019, columns=labels_2019)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_df_2019, annot=True, fmt='d', cmap='YlGnBu')
plt.title("2019 Confusion Matrix")
plt.ylabel("True Region")
plt.xlabel("Predicted Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/confusion_matrix_2019.png")
plt.close()

print("✅ Pipeline complete! Outputs saved in ./output and ./models/")