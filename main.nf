nextflow.enable.dsl=2

// =========================
// PROCESS DEFINITIONS
// =========================

process region_rf_pipeline {
    tag "Train and Predict RF"

    input:
        path train_meta
        path train_kmers
        path test_meta
        path test_kmers
        path script

    output:
        path "models"
        path "output"

    script:
    """
    mkdir -p models output
    python3 ${script}
    """
}

process blast_top10 {
    tag "BLAST top kmers"

    input:
        path top10_csv

    output:
        path "blast_results"

    script:
    """
    mkdir -p blast_results
    bash scripts/blast_top10.sh
    """
}

// =========================
// WORKFLOW BLOCK
// =========================

workflow {

    // Declare inputs
    train_meta   = file(params.train_meta)
    train_kmers  = file(params.train_kmers)
    test_meta    = file(params.test_meta)
    test_kmers   = file(params.test_kmers)
    script       = file(params.script)

    // Step 1: train + predict
    rf_output = region_rf_pipeline(
        train_meta,
        train_kmers,
        test_meta,
        test_kmers,
        script
    )

    // Step 2: run BLAST on top 10 kmers after RF step
    top10_csv = rf_output.out[1].filter { it.name.endsWith("top10_kmers.csv") }
    blast_top10(top10_csv)
} 
