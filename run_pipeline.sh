#!/bin/bash
set -e

ENV_NAME="rf-region-classifier"

echo "\ud83d\udce6 Checking if Conda environment '$ENV_NAME' exists..."
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "\u2705 Conda environment '$ENV_NAME' already exists."
else
    echo "\ud83d\udd27 Creating conda environment from environment.yml..."
    conda env create -f environment.yml
fi

echo "\ud83d\udcc2 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "\ud83d\ude80 Running pipeline via Nextflow..."
nextflow run main.nf

