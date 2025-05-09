#!/bin/bash
set -euo pipefail

ENV_NAME="rf-stec-classifier" 
START_TIME=$(date +%s)
RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "==============================================="
echo "         RF Region & Country Pipeline"
echo "==============================================="
echo "📅 Start time : $(date)"
echo "🔍 Conda env  : $ENV_NAME"
echo "🆔 Run ID     : $RUN_ID"
echo "==============================================="

# === Check for Conda ===
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# === Conda Environment Setup ===
echo "📦 Checking if Conda environment '$ENV_NAME' exists..."
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "✅ Conda environment '$ENV_NAME' already exists."
else
    echo "🔧 Creating conda environment from environment.yml..."
    conda env create -f environment.yml -n "$ENV_NAME"
fi

# === Activate Conda ===
echo "📂 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# === Run Nextflow Pipeline ===
echo "🚀 Running Nextflow pipeline..."
nextflow run main.nf \
    -with-conda \
    -with-report report_${RUN_ID}.html \
    -with-trace trace_${RUN_ID}.txt \
    -resume \
    "$@"

# === Wrap Up ===
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "==============================================="
echo "🎉 Pipeline execution completed successfully!"
echo "==============================================="
echo "📁 Outputs generated:"
echo "   - Cleaned metadata        : data/Training/metadata_14_18_cleaned.csv"
echo "   - Region model outputs    : output/"
echo "   - Country model outputs   : output_country/"
echo "   - Region BLAST results    : blast_results/"
echo "   - Country BLAST results   : blast_results_country/"
echo "==============================================="
echo "📊 Pipeline reports:"
echo "   - Execution report : report_${RUN_ID}.html"
echo "   - Execution trace  : trace_${RUN_ID}.txt"
echo "==============================================="
echo "⏱️ Total runtime : ${RUNTIME} seconds ($(echo "$RUNTIME / 60" | bc) min)"
echo "📅 End time     : $(date)"
echo "==============================================="
