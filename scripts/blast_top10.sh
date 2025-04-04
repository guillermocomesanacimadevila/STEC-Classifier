#!/bin/bash
set -e

INPUT="output/top10_kmers.csv"
BLAST_DIR="blast_results"
mkdir -p "$BLAST_DIR"

if ! command -v blastn &> /dev/null; then
    echo "❌ BLAST+ not installed or not in PATH. Please install BLAST (e.g., via conda)."
    exit 1
fi

echo "🧬 Running BLAST for top 10 kmers..."

# Skip header, clean input, and loop
tail -n +2 "$INPUT" | while IFS=, read -r kmer importance; do
    # Clean kmer (remove quotes/spaces)
    kmer_clean=$(echo "$kmer" | tr -d '\r\n\"' | sed 's/[^ATCGNatcgn]//g')
    if [[ -z "$kmer_clean" ]]; then continue; fi

    echo ">${kmer_clean}" > temp_kmer.fa
    echo "${kmer_clean}" >> temp_kmer.fa

    blastn -db nt -remote -query temp_kmer.fa -outfmt 6 -max_target_seqs 1 \
        -out "${BLAST_DIR}/${kmer_clean}_blast.txt" || echo "⚠️ BLAST failed for $kmer_clean"

    rm -f temp_kmer.fa
done

echo "✅ BLAST complete. Results in $BLAST_DIR/"