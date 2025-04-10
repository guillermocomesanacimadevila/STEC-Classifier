#!/bin/bash
set -euo pipefail

ENV_NAME="rf-region-classifier"

INPUT="top10_kmers.csv"
BLAST_DIR="blast_results"
SUMMARY="${BLAST_DIR}/blast_summary.txt"

mkdir -p "$BLAST_DIR"
> "$SUMMARY"

start_time=$(date +%s)

echo "🧬 Starting BLAST on top 10 kmers from: $INPUT"
echo "🔽 Output folder: $BLAST_DIR"

# Check BLAST+ installation
if ! command -v blastn &> /dev/null; then
    echo "❌ BLAST+ not installed or not in PATH."
    echo "   Install via: conda install -c bioconda blast"
    exit 1
fi

# Skip header and process each k-mer
tail -n +2 "$INPUT" | while IFS=, read -r kmer importance; do
    kmer_clean=$(echo "$kmer" | tr -d '\r\n\"' | sed 's/[^ATCGNatcgn]//g')

    if [[ -z "$kmer_clean" ]]; then
        echo "⚠️ Skipping empty/invalid k-mer: '$kmer'"
        continue
    fi

    fasta_file=$(mktemp)
    echo ">kmer_${kmer_clean}" > "$fasta_file"
    echo "$kmer_clean" >> "$fasta_file"

    output_file="${BLAST_DIR}/${kmer_clean}_blast.txt"
    echo "🔍 BLASTing k-mer: $kmer_clean..."

    if blastn -db nt -remote -query "$fasta_file" -outfmt 6 -max_target_seqs 1 -out "$output_file"; then
        hit=$(head -n 1 "$output_file" || echo "No hits")
        echo -e "${kmer_clean}\t${importance}\t${hit}" >> "$SUMMARY"
    else
        echo "❌ BLAST failed for: $kmer_clean" | tee -a "$SUMMARY"
    fi

    rm -f "$fasta_file"
done

end_time=$(date +%s)
runtime=$((end_time - start_time))

echo "✅ BLAST complete. Summary saved to: $SUMMARY"
echo "🕒 Total runtime: ${runtime} seconds"