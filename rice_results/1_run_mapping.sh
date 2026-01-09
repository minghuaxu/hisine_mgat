#!/bin/bash

# 参数设置
REF_GENOME="/home/xuminghua/data/genomes/plants/Oryza_sativa-Japanese_rice/GCF_034140825.1_ASM3414082v1_genomic.fna"    # 水稻参考基因组
REPBASE_FASTA="/home/xuminghua/data/databases/repbase/RepBase_SINE/oryrep.ref" # Repbase中的SINE序列
MY_SINE_FASTA="/home/xuminghua/sine_classifier/sine_classifier/predictions_refined_clustered"      # 你模型识别的SINE序列
THREADS=10

# 1. 建库 (如果尚未建库)
if [ ! -f "${REF_GENOME}.nsq" ]; then
    echo "[INFO] Building BLAST database for reference genome..."
    makeblastdb -in "${REF_GENOME}" -dbtype nucl -out rice_ref_db
else
    echo "[INFO] BLAST database already exists."
fi

# 定义输出格式: standard 6 columns + qlen (query length)
# 格式: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen
OUTFMT="6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen"

# 2. Repbase 比对到参考基因组 (Ground Truth)
echo "[INFO] Mapping Repbase to Reference Genome..."
blastn \
    -query "${REPBASE_FASTA}" \
    -db rice_ref_db \
    -out repbase_to_genome.out \
    -evalue 1e-10 \
    -outfmt "${OUTFMT}" \
    -num_threads ${THREADS}

# 3. 预测序列 比对到参考基因组 (Prediction)
echo "[INFO] Mapping My SINEs to Reference Genome..."
blastn \
    -query "${MY_SINE_FASTA}" \
    -db rice_ref_db \
    -out pred_to_genome.out \
    -evalue 1e-10 \
    -outfmt "${OUTFMT}" \
    -num_threads ${THREADS}

echo "[INFO] Mapping finished."