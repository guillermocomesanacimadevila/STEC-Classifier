#!/usr/bin/env python3
# data_cleaning_upgraded_v2.py

import os
import argparse
import pandas as pd

def load_metadata(train_path, test_path):
    for path in [train_path, test_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"❌ File not found: {path}")

    metadata_tr = pd.read_csv(train_path)
    metadata_te = pd.read_csv(test_path)

    if 'Accession' not in metadata_tr.columns or 'Accession' not in metadata_te.columns:
        raise KeyError("❌ 'Accession' column not found in one of the input files.")

    metadata_tr.set_index("Accession", inplace=True)
    metadata_te.set_index("Accession", inplace=True)

    return metadata_tr, metadata_te

def normalize_text_fields(df):
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("").astype(str)
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        df[col] = df[col].str.replace(r'\.$', '', regex=True)
        df[col] = df[col].str.title()
    return df

def clean_country_column(df):
    country_map = {
        'N': 'UK', 'Wales': 'UK', 'Scotland': 'UK', 'England': 'UK',
        'Portgual': 'Portugal', 'Saudia Arabia': 'Saudi Arabia',
        'Holland': 'Netherlands', 'USA': 'United States',
        'U.S.A.': 'United States', 'United States Of America': 'United States',
        'Republic Of Ireland': 'Ireland', 'Northen Ireland': 'UK',
        'Philipines': 'Philippines'
    }
    df['Country'] = df['Country'].replace(country_map)
    return df

def clean_region_column(df):
    region_map = {
        'S. America': 'South America',
        'N. America': 'North America',
        'C. Europe': 'Central Europe'
    }
    df['Region'] = df['Region'].replace(region_map)
    return df

def clean_stx_column(df):
    df['Stx'] = df['Stx'].str.strip()
    df = df[df['Stx'].notna() & ~df['Stx'].str.lower().isin(['none', 'unknown', 'nan'])]
    return df

def clean_pt_column(df):
    df['PT'] = df['PT'].str.strip()
    df = df[df['PT'].notna() & ~df['PT'].str.lower().isin(['none', 'unknown', 'nan'])]
    return df

def remove_duplicates(df):
    before = df.shape[0]
    df = df[~df.index.duplicated(keep='first')]
    after = df.shape[0]
    print(f"\n🧹 Removed {before - after} duplicate accession entries.")
    return df

def validate_country_region_alignment(df):
    mismatch = df.groupby('Country')['Region'].nunique()
    inconsistent = mismatch[mismatch > 1]
    if not inconsistent.empty:
        print("\n❗ Inconsistent Region assignments for some countries:")
        print(inconsistent)

def report_unseen_classes(train_df, test_df, field="Country"):
    train_classes = set(train_df[field].unique())
    test_classes = set(test_df[field].unique())
    unseen = test_classes - train_classes
    if unseen:
        print(f"\n⚠️ {field} classes only in test set ({len(unseen)}): {sorted(unseen)}")

def remove_unseen_test_classes(train_df, test_df, fields=["Country", "Region"]):
    for field in fields:
        seen_labels = set(train_df[field].unique())
        original_count = len(test_df)
        test_df = test_df[test_df[field].isin(seen_labels)]
        removed = original_count - len(test_df)
        if removed > 0:
            print(f"⚠️ Removed {removed} test samples with unseen {field} labels.")
    return test_df

def clean_metadata(metadata_tr, metadata_te):
    metadata_tr = normalize_text_fields(metadata_tr)
    metadata_te = normalize_text_fields(metadata_te)

    metadata_tr = clean_country_column(metadata_tr)
    metadata_te = clean_country_column(metadata_te)

    metadata_tr = clean_region_column(metadata_tr)
    metadata_te = clean_region_column(metadata_te)

    metadata_tr = clean_stx_column(metadata_tr)
    metadata_te = clean_stx_column(metadata_te)

    metadata_tr = clean_pt_column(metadata_tr)
    metadata_te = clean_pt_column(metadata_te)

    metadata_tr = remove_duplicates(metadata_tr)
    metadata_te = remove_duplicates(metadata_te)

    validate_country_region_alignment(metadata_tr)
    report_unseen_classes(metadata_tr, metadata_te, field="Country")
    report_unseen_classes(metadata_tr, metadata_te, field="Region")

    print(f"\n✅ Cleaned training shape: {metadata_tr.shape}")
    print(f"✅ Cleaned testing shape: {metadata_te.shape}")

    return metadata_tr, metadata_te

def filter_by_kmers(df, df2, kmer_train_path, kmer_test_path):
    kmer_train_ids = pd.read_csv(kmer_train_path, sep="\t", nrows=1).columns.tolist()[1:]
    kmer_test_ids = pd.read_csv(kmer_test_path, sep="\t", nrows=1).columns.tolist()[1:]

    df_filtered = df[df.index.isin(kmer_train_ids)]
    df2_filtered = df2[df2.index.isin(kmer_test_ids)]

    print(f"\n✅ Training metadata filtered to {len(df_filtered)} samples matching k-mer file.")
    print(f"✅ Test metadata filtered to {len(df2_filtered)} samples matching k-mer file.")

    shared = set(df.index) & set(kmer_train_ids)
    only_in_meta = set(df.index) - set(kmer_train_ids)
    only_in_kmer = set(kmer_train_ids) - set(df.index)
    print(f"\n🔍 Sample ID overlap diagnostic:")
    print(f"Shared samples: {len(shared)}")
    print(f"Only in metadata: {len(only_in_meta)} → {list(only_in_meta)[:5]}")
    print(f"Only in k-mer file: {len(only_in_kmer)} → {list(only_in_kmer)[:5]}")

    return df_filtered, df2_filtered

def print_summary(df, df2, original_train, original_test):
    print("\n=== Summary ===")
    print("Training set missing values:\n", df.isna().sum())
    print("Testing set missing values:\n", df2.isna().sum())

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
    print(f"Original training shape: {original_train.shape} → Cleaned: {df.shape}")
    print(f"Original test shape: {original_test.shape} → Cleaned: {df2.shape}")

    print("\n✅ Example cleaned training sample IDs:")
    print(df.index[:5])

def save_cleaned_metadata(df, df2, output_dir="data"):
    os.makedirs(f"{output_dir}/Training", exist_ok=True)
    os.makedirs(f"{output_dir}/Testing", exist_ok=True)

    df.to_csv(f"{output_dir}/Training/metadata_14_18_cleaned.csv", sep=",", index=True, index_label="Accession")
    df2.to_csv(f"{output_dir}/Testing/metadata_19_cleaned.csv", sep=",", index=True, index_label="Accession")
    print("\n✅ Metadata cleaning complete. Cleaned files saved.")

def main():
    parser = argparse.ArgumentParser(description="Clean metadata files and filter by k-mer sample IDs.")
    parser.add_argument("--train_metadata", required=True, help="Path to raw training metadata")
    parser.add_argument("--test_metadata", required=True, help="Path to raw test metadata")
    parser.add_argument("--train_kmers", required=True, help="Path to training k-mer file")
    parser.add_argument("--test_kmers", required=True, help="Path to test k-mer file")
    parser.add_argument("--output_dir", default="data", help="Directory to save cleaned outputs")
    args = parser.parse_args()

    metadata_tr, metadata_te = load_metadata(args.train_metadata, args.test_metadata)
    cleaned_tr, cleaned_te = clean_metadata(metadata_tr, metadata_te)
    cleaned_te = remove_unseen_test_classes(cleaned_tr, cleaned_te)
    filtered_tr, filtered_te = filter_by_kmers(cleaned_tr, cleaned_te, args.train_kmers, args.test_kmers)
    print_summary(filtered_tr, filtered_te, metadata_tr, metadata_te)
    save_cleaned_metadata(filtered_tr, filtered_te, args.output_dir)

if __name__ == "__main__":
    main()
