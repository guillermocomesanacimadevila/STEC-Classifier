#!/usr/bin/env python3
# 01_data_cleaning.py

import os
import sys
import pandas as pd

# === Load Metadata from command-line arguments ===
if len(sys.argv) != 3:
    print("Usage: python 01_data_cleaning.py <train_csv> <test_csv>")
    sys.exit(1)

train_path = sys.argv[1]
test_path = sys.argv[2]

# === Check that input files exist ===
if not os.path.isfile(train_path):
    print(f"❌ Training file not found: {train_path}")
    sys.exit(2)

if not os.path.isfile(test_path):
    print(f"❌ Test file not found: {test_path}")
    sys.exit(3)

# === Read files ===
metadata_tr = pd.read_csv(train_path, sep=",", index_col=0)
metadata_te = pd.read_csv(test_path, sep=",", index_col=0)

# === Country Fixes ===
metadata_tr['Country'] = metadata_tr['Country'].replace({
    'N': 'UK',
    'Portgual': 'Portugal',
    'Wales': 'UK'
})
metadata_te['Country'] = metadata_te['Country'].replace({
    'N': 'UK'
})

# === Clean 'Stx' Column ===
df = metadata_tr[
    metadata_tr["Stx"].astype(str).str.strip().str.lower() != "none"
]
df = df[df["Stx"].notna()]
print(f"Stx cleaned: {metadata_tr.shape[0] - df.shape[0]} entries removed")

# === Clean 'PT' Column in Training Set ===
df = df[
    df["PT"].astype(str).str.strip().str.lower() != "nan"
]
df = df[df["PT"].notna()]
print(f"Training data shape after cleaning: {df.shape}")

# === Clean 'PT' and 'Stx' in Test Set ===
df2 = metadata_te[
    (metadata_te["PT"].notna()) &
    (metadata_te["PT"].astype(str).str.strip().str.lower() != "nan")
]
df2 = df2[
    (df2["Stx"].notna()) &
    (df2["Stx"].astype(str).str.strip().str.lower() != "none")
]

# === Summary ===
print("\n=== Summary ===")
print("Training set missing values:")
print(df.isna().sum())
print("Testing set missing values:")
print(df2.isna().sum())

print("\nUnique values per field (Training):")
print(f"Countries: {df['Country'].nunique()}")
print(f"Regions: {df['Region'].nunique()}")
print(f"Stx types: {df['Stx'].nunique()}")
print(f"PT types: {df['PT'].nunique()}")

print("\nUnique values per field (Testing):")
print(f"Countries: {df2['Country'].nunique()}")
print(f"Regions: {df2['Region'].nunique()}")
print(f"Stx types: {df2['Stx'].nunique()}")
print(f"PT types: {df2['PT'].nunique()}")

print("\nShape comparison:")
print(f"Original training shape: {metadata_tr.shape} → Cleaned: {df.shape}")
print(f"Original test shape: {metadata_te.shape} → Cleaned: {df2.shape}")

# === Save cleaned dfs into local (dynamic) working directory ===
base_dir = os.getcwd()
os.makedirs(f"{base_dir}/data/Training", exist_ok=True)
os.makedirs(f"{base_dir}/data/Testing", exist_ok=True)

df.to_csv(f"{base_dir}/data/Training/metadata_14_18_cleaned.csv", sep=",", index=True)
df2.to_csv(f"{base_dir}/data/Testing/metadata_19_cleaned.csv", sep=",", index=True)

print("\n✅ Metadata cleaning complete. Cleaned files saved.")