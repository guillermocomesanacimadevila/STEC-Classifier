# Base image
FROM continuumio/miniconda3:latest

# Metadata
LABEL maintainer="Your Name <your.email@institution.org>" \
      description="Dockerfile for STEC geographic classification using Random Forest"

# Set working directory
WORKDIR /app

# Copy environment file (optional if you have one)
# COPY environment.yml .

# Create Conda environment and install Python
RUN conda update -n base -c defaults conda && \
    conda install -y python=3.12 && \
    conda install -y -c conda-forge \
        numpy=1.26.4 \
        pandas=2.2.2 \
        matplotlib=3.8.4 \
        seaborn=0.13.2 \
        scikit-learn=1.4.1 \
        joblib=1.4.0 \
        imbalanced-learn=0.12.1 \
        nextflow=22.10.1 && \
    conda clean -afy

# Install NCBI BLAST+ from Bioconda
RUN conda install -y -c bioconda blast=2.13.0 && \
    conda clean -afy

# Install other dependencies (if needed)
RUN apt-get update && \
    apt-get install -y wget curl bash git && \
    rm -rf /var/lib/apt/lists/*

# Copy pipeline scripts (assuming you have these in your build context)
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TZ=UTC

# Set entrypoint (optional)
ENTRYPOINT ["/bin/bash"]
