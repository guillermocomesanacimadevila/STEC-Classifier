nextflow.enable.dsl=2

// ==================== PARAMETERS ==================== //

params.train_metadata = "data/Training/14-18metadata"
params.test_metadata  = "data/Testing/19metadata"
params.train_kmers    = "data/Training/14-18kmerdata.txt"
params.test_kmers     = "data/Testing/19kmerdata.txt"
params.outdir         = "results"
params.rf_script      = "scripts/02_rf_region_pipeline.py"
params.cleaning_script = "scripts/01_data_cleaning.py"
params.blast_script    = "scripts/03_blast_top10.sh"
params.target_column   = "Region"

// ==================== CHANNELS ==================== //

Channel.fromPath(params.train_metadata)  .set { train_metadata_ch }
Channel.fromPath(params.test_metadata)   .set { test_metadata_ch }
Channel.fromPath(params.train_kmers)     .set { train_kmers_ch }
Channel.fromPath(params.test_kmers)      .set { test_kmers_ch }

Channel.value(file(params.cleaning_script)).set { cleaning_script_ch }
Channel.value(file(params.rf_script)).set { rf_script_ch }
Channel.value(file(params.blast_script)).set { blast_script_ch }

// ==================== WORKFLOW ==================== //

workflow {

    // Step 1: Clean metadata
    clean_metadata(train_metadata_ch.combine(test_metadata_ch).combine(cleaning_script_ch))

    // Step 2: Train RF model using cleaned metadata + kmer files
    region_rf_pipeline(
        clean_metadata.out.combine(train_kmers_ch)
                          .combine(test_kmers_ch)
                          .combine(rf_script_ch)
    )

    // Step 3: BLAST top 10 kmers
    blast_top10(
        region_rf_pipeline.out.combine(blast_script_ch)
    )
}

// ==================== PROCESSES ==================== //

process clean_metadata {

    tag "Cleaning Metadata"

    conda = './environment.yml'

    input:
    tuple path(train_metadata), path(test_metadata), path(script_file)

    output:
    tuple path("data/Training/metadata_14_18_cleaned.csv"),
          path("data/Testing/metadata_19_cleaned.csv")

    script:
    """
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate rf-region-classifier
    python ${script_file} ${train_metadata} ${test_metadata}
    """
}

process region_rf_pipeline {

    tag "Random Forest Region Classification"

    conda = './environment.yml'

    input:
    tuple path(train_metadata_cleaned), path(test_metadata_cleaned),
          path(train_kmers), path(test_kmers),
          path(script_file)

    output:
    path "output/top10_kmers.csv"

    script:
    """
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate rf-region-classifier
    python ${script_file} \\
        --train_metadata ${train_metadata_cleaned} \\
        --test_metadata ${test_metadata_cleaned} \\
        --train_kmers ${train_kmers} \\
        --test_kmers ${test_kmers} \\
        --output_dir output \\
        --model_dir models \\
        --target_column ${params.target_column}
    """
}

process blast_top10 {

    tag "BLAST Top 10 kmers"

    conda = './environment.yml'

    input:
    tuple path(top10_kmers), path(script_file)

    output:
    path "blast_results/blast_summary.txt"

    publishDir "${params.outdir}/blast", mode: 'copy'

    script:
    """
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate rf-region-classifier
    bash ${script_file}
    """
}
