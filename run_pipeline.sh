#!/bin/bash
set -euo pipefail

ENV_NAME="rf-region-classifier"
START_TIME=$(date +%s)
RUN_ID=$(date +%Y%m%d_%H%M%S)

echo "==============================================="
echo "         RF Region Classifier Pipeline"
echo "==============================================="
echo "📅 Start time : $(date)"
echo "🔍 Conda env  : $ENV_NAME"
echo "==============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check Conda environment
echo "📦 Checking if Conda environment '$ENV_NAME' exists..."
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "✅ Conda environment '$ENV_NAME' already exists."
else
    echo "🔧 Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

echo "📂 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "🚀 Running Nextflow pipeline..."
nextflow run main.nf -with-conda -with-report report_${RUN_ID}.html -with-trace trace_${RUN_ID}.txt "$@"

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))

echo "==============================================="
echo "🎉 Pipeline execution completed successfully!"
echo "==============================================="
echo "📁 Outputs generated:"
echo "   - Cleaned metadata      : data/Training/metadata_14_18_cleaned.csv"
echo "   - Trained model outputs : models/"
echo "   - RF evaluation results : output/"
echo "   - BLAST results         : blast_results/ or results/blast/"
echo "==============================================="
echo "📊 Pipeline reports:"
echo "   - Execution report : report_${RUN_ID}.html"
echo "   - Execution trace  : trace_${RUN_ID}.txt"
echo "==============================================="
echo "⏱️ Total runtime : ${RUNTIME} seconds ($(echo "$RUNTIME / 60" | bc) min)"
echo "📅 End time     : $(date)"
echo "==============================================="
