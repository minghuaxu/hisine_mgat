# -*- coding: utf-8 -*-
"""
SINE Classifier Snakemake Workflow - 端到端版本
=================================================

关键改动:
1. ❌ 删除了 generate_embeddings 规则
2. ❌ 删除了 aggregate_embeddings 规则
3. ✅ 训练直接使用FASTA + Motif坐标
4. ✅ 简化的流程，更清晰
"""
import sys
import os
sys.path.insert(0, '/homeb/xuminghua/hisine_classifier')

configfile: "config/config.yaml"

SPECIES_LIST = list(config["samples"].keys())

# ============================================================================
# 最终目标
# ============================================================================

rule all:
    input:
        "results/model/e2e_finetuned/final_model/best_model.pt",
        # 预测任务（可选）
        # expand("results/predictions/{sample_id}/{sample_id}_predictions.tsv", 
        #        sample_id=config["prediction"]["samples"])


# ============================================================================
# 步骤1: 提取正负样本
# ============================================================================

rule extract_positives:
    input:
        genome=lambda wildcards: config["samples"][wildcards.species]["genome"],
        sine_ref=config["sine_ref"],
        # 需要其他的库来做交叉验证
        dna_ref=config["negative_refs"]["DNA"],
        ltr_ref=config["negative_refs"]["LTR"],
        tir_ref=config["negative_refs"]["TIR"]
    output:
        tsv="results/{species}/positives/samples.tsv",
        fa="results/{species}/positives/samples.fa"
    params:
        # 先生成到临时文件，清洗后再覆盖到 output
        raw_prefix="results/{species}/positives/raw_samples",
        final_prefix="results/{species}/positives/samples"
    threads: config["threads"]
    log: "logs/{species}/extract_positives.log"
    shell:
        """
        # 1. 原始提取 (输出到 raw_samples.tsv/.fa)
        python scripts/01_extract_positives.py \
            --genome {input.genome} \
            --sine_ref {input.sine_ref} \
            --out_prefix {params.raw_prefix} \
            --threads {threads} \
            --flank {config[flank_size]} \
            --cov_thr {config[min_coverage_ratio]} \
            --min_as_score 50 \
            --max_de_divergence 0.25 >> {log} 2>&1
        
        echo "------------------------------------------------" >> {log}
        echo "开始交叉比对清洗 (Cross-Check Contamination)..." >> {log}

        # 2. 交叉清洗 (输入 raw，输出到 final: samples.tsv/.fa)
        python tools/remove_contamination.py \
            --input_fa {params.raw_prefix}.fa \
            --input_tsv {params.raw_prefix}.tsv \
            --neg_refs {input.dna_ref} {input.ltr_ref} {input.tir_ref} \
            --out_prefix {params.final_prefix} \
            --threads {threads} \
            --cov_thr 0.6 >> {log} 2>&1
            
        # 3. 清理 raw 文件 (可选)
        # rm {params.raw_prefix}.fa {params.raw_prefix}.tsv {params.raw_prefix}.sam
        """


rule extract_negatives:
    input:
        genome=lambda wildcards: config["samples"][wildcards.species]["genome"],
        gff=lambda wildcards: config["samples"][wildcards.species]["gff"],
        pos_tsv="results/{species}/positives/samples.tsv",
        sine_ref=config["sine_ref"],
        dna_ref=config["negative_refs"]["DNA"],
        ltr_ref=config["negative_refs"]["LTR"],
        tir_ref=config["negative_refs"]["TIR"],
    output:
        tsv="results/{species}/negatives/samples.tsv",
        fa="results/{species}/negatives/samples.fa"
    params:
        prefix="results/{species}/negatives/samples",
        ratio_bg=config["negative_ratios"]["bg"]
    threads: config["threads"]
    log: "logs/{species}/extract_negatives.log"
    shell:
        """
        if [ -s {input.pos_tsv} ]; then
            pos_count=$(tail -n +2 {input.pos_tsv} | wc -l)
        else
            pos_count=0
        fi

        python scripts/02_extract_negatives.py \
            --genome {input.genome} \
            --gff {input.gff} \
            --pos_tsv {input.pos_tsv} \
            --pos_count $pos_count \
            --sine_ref {input.sine_ref} \
            --dna_ref {input.dna_ref} \
            --ltr_ref {input.ltr_ref} \
            --tir_ref {input.tir_ref} \
            --ratio_bg {params.ratio_bg} \
            --out_prefix {params.prefix} \
            --flank {config[flank_size]} \
            --threads {threads} \
            --cov_thr {config[min_coverage_ratio]} \
            --max_de_divergence 0.25 > {log} 2>&1
        """


# ============================================================================
# 步骤2: 检测Motif特征（不生成embedding）
# ============================================================================

rule find_motifs:
    input:
        "results/{species}/{sample_type}/samples.tsv"
    output:
        unified="results/{species}/{sample_type}/motifs.unified_coordinates.tsv"
    params:
        prefix="results/{species}/{sample_type}/motifs"
    log: "logs/{species}/find_motifs_{sample_type}.log"
    shell:
        """
        python scripts/03_detect_motifs.py \
            --in_tsv {input} \
            --out_prefix {params.prefix} \
            --new_flank_len 100 > {log} 2>&1
        """


rule all_motif_extraction:
    """任务: 提取所有物种的Motif特征"""
    input:
        expand("results/{species}/{sample_type}/motifs.unified_coordinates.tsv", 
               species=SPECIES_LIST, 
               sample_type=["positives", "negatives"])
    shell:
        "echo '✅ 所有物种的Motif特征提取完成'"


# ============================================================================
# 步骤3: 合并训练数据（仅FASTA和Motif坐标）
# ============================================================================

rule combine_fasta_for_e2e:
    input:
        pos_fastas=expand("results/{species}/positives/samples.fa", species=SPECIES_LIST),
        neg_fastas=expand("results/{species}/negatives/samples.fa", species=SPECIES_LIST)
    output:
        "results/all_species_aggregated/e2e_training_data.fa"
    log: "logs/combine_fastas_e2e.log"
    shell:
        """
        python tools/combine_training_data.py \
            --pos_fastas {input.pos_fastas} \
            --neg_fastas {input.neg_fastas} \
            --output_fasta {output} > {log} 2>&1
        """

# 使用 CD-HIT 进行聚类划分，防止数据泄露
rule cluster_split_data:
    input:
        fasta = "results/all_species_aggregated/e2e_training_data.fa"
    output:
        train_ids = "results/data_split/train_ids.txt",
        val_ids = "results/data_split/val_ids.txt",
        # 标记文件，确保聚类完成
        done = "results/data_split/clustering.done"
    params:
        out_dir = "results/data_split",
        threshold = 0.8, # 相似度阈值
        val_ratio = 0.2  # 验证集比例
    threads: 32
    log: "logs/cluster_split_data.log"
    shell:
        """
        python tools/split_dataset_by_cluster.py \
            --input_fasta {input.fasta} \
            --out_dir {params.out_dir} \
            --val_ratio {params.val_ratio} \
            --threshold {params.threshold} \
            --threads {threads} > {log} 2>&1
        
        touch {output.done}
        """

rule aggregate_motif_coords_for_e2e:
    input:
        coords_tsvs=expand("results/{species}/{sample_type}/motifs.unified_coordinates.tsv", 
                           species=SPECIES_LIST, 
                           sample_type=["positives", "negatives"])
    output:
        "results/all_species_aggregated/all_motif_coordinates.tsv"
    log: "logs/aggregate_motif_coords_e2e.log"
    shell:
        """
        (head -n 1 {input.coords_tsvs[0]} && tail -n +2 -q {input.coords_tsvs}) > {output}
        echo "✅ 已聚合 $(wc -l < {output}) 行motif坐标" >> {log}
        """


# ============================================================================
# 步骤4: 端到端训练（不需要预计算embedding）
# ============================================================================

rule train_e2e_sine_classifier:
    input:
        training_fasta="results/all_species_aggregated/e2e_training_data.fa",
        motif_coords="results/all_species_aggregated/all_motif_coordinates.tsv",
        backbone=config["e2e_training"]["plant_backbone_path"],
        split_done="results/data_split/clustering.done"
    output:
        final_model="results/model/e2e_finetuned/final_model/best_model.pt"
    params:
        output_dir="results/model/e2e_finetuned/final_model",
        split_dir="results/data_split", 
        num_gpus=config["e2e_training"]["num_gpus"],
        gpu_env_prefix=(
            f"CUDA_VISIBLE_DEVICES={config['e2e_training']['gpu_ids']}"
            if config["e2e_training"].get("gpu_ids") 
            else ""
        ),
        epochs=config["e2e_training"]["epochs"],
        freeze_epochs=config["e2e_training"]["freeze_epochs"],       # 前 n-1 轮冻结，第 n 轮开始解冻
        batch_size=config["e2e_training"]["batch_size"],
        backbone_lr=config["e2e_training"]["backbone_lr"],
        head_lr=config["e2e_training"]["head_lr"],
        hidden_dim=config["e2e_training"]["hidden_dim"],
        dropout=config["e2e_training"]["dropout"],
    threads: 32
    resources:
        gpu=config["e2e_training"]["num_gpus"]
    log: "logs/train_e2e_model.log"
    shell:
        """
        # 构建训练命令
        freeze_flag=""
        if [ "{params.freeze_backbone}" = "true" ]; then
            freeze_flag="--freeze_backbone"
        fi
        
        {params.gpu_env_prefix} NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node={params.num_gpus} \
        scripts/04_train_e2e_classifier.py \
            --backbone_path {input.backbone} \
            --sine_data_path {input.training_fasta} \
            --motif_data_path {input.motif_coords} \
            --output_dir {params.output_dir} \
            --split_dir {params.split_dir} \
            --epochs {params.epochs} \
            --freeze_epochs {params.freeze_epochs} \
            --batch_size {params.batch_size} \
            --backbone_lr {params.backbone_lr} \
            --head_lr {params.head_lr} \
            --hidden_dim {params.hidden_dim} \
            --dropout {params.dropout} 
            > {log} 2>&1
        """


# ============================================================================
# 步骤5: 预测新序列
# ============================================================================

rule all_prediction:
    input:
        expand("results/predictions/{sample_id}/{sample_id}_predictions.tsv", 
               sample_id=config["prediction"]["samples"]),
        expand("results/predictions/{sample_id}/{sample_id}_predicted_SINEs.fa",
               sample_id=config["prediction"]["samples"])
    shell:
        "echo '✅ 所有预测任务完成'"


rule preprocess_for_prediction:
    input:
        tsv=lambda wildcards: config["prediction"]["inputs"][wildcards.sample_id]
    output:
        fasta="results/predictions/{sample_id}/preprocessed.fa",
        standard_tsv="results/predictions/{sample_id}/preprocessed.tsv"
    log: "logs/predictions/preprocess_{sample_id}.log"
    shell:
        """
        python tools/preprocess_for_prediction.py \
            --input_tsv {input.tsv} \
            --output_fasta {output.fasta} \
            --output_tsv {output.standard_tsv} > {log} 2>&1
        """


rule find_motifs_for_prediction:
    input:
        "results/predictions/{sample_id}/preprocessed.tsv"
    output:
        unified="results/predictions/{sample_id}/motifs.unified_coordinates.tsv"
    params:
        prefix="results/predictions/{sample_id}/motifs"
    log: "logs/predictions/find_motifs_{sample_id}.log"
    shell:
        """
        python scripts/03_detect_motifs.py \
            --in_tsv {input} \
            --out_prefix {params.prefix} \
            --fast > {log} 2>&1  # <--- 加上 --fast
        """


rule run_prediction:
    input:
        model=config["prediction"]["trained_model"],
        backbone=config["e2e_training"]["plant_backbone_path"],
        full_fasta="results/predictions/{sample_id}/preprocessed.fa",
        motif_coords="results/predictions/{sample_id}/motifs.unified_coordinates.tsv",
        preprocessed_tsv="results/predictions/{sample_id}/preprocessed.tsv"
    output:
        tsv="results/predictions/{sample_id}/{sample_id}_predictions.tsv",
        positive_fasta="results/predictions/{sample_id}/{sample_id}_predicted_SINEs.fa"
    params:
        prefix="results/predictions/{sample_id}/{sample_id}",
        hidden_dim=config["e2e_training"]["hidden_dim"],
        dropout=config["e2e_training"]["dropout"]
    log: "logs/predictions/run_{sample_id}.log"
    shell:
        """
        python scripts/05_predict.py \
            --model_path {input.model} \
            --backbone_path {input.backbone} \
            --preprocessed_tsv {input.preprocessed_tsv} \
            --input_fasta {input.full_fasta} \
            --motif_coords {input.motif_coords} \
            --output_prefix {params.prefix} \
            --hidden_dim {params.hidden_dim} \
            --dropout {params.dropout} > {log} 2>&1
        """


# ============================================================================
# 辅助任务
# ============================================================================

rule test_alignment:
    """测试offset对齐是否正确"""
    output:
        "results/validation/alignment_test.passed"
    log: "logs/test_alignment.log"
    shell:
        """
        python tools/validate_alignment.py > {log} 2>&1
        touch {output}
        echo "✅ Offset对齐验证通过" >> {log}
        """


rule visualize_samples:
    """可视化前N个训练样本的token对齐"""
    input:
        training_fasta="results/all_species_aggregated/e2e_training_data.fa",
        motif_coords="results/all_species_aggregated/all_motif_coordinates.tsv"
    output:
        "results/visualization/sample_alignment_{n}.png"
    params:
        sample_idx=lambda wildcards: int(wildcards.n)
    log: "logs/visualize_samples_{n}.log"
    shell:
        """
        python tools/visualize_training_samples.py \
            --training_fasta {input.training_fasta} \
            --motif_coords {input.motif_coords} \
            --sample_idx {params.sample_idx} \
            --output {output} > {log} 2>&1
        """


rule check_model_gradients:
    """检查模型梯度是否正确传播"""
    input:
        backbone=config["e2e_training"]["plant_backbone_path"]
    output:
        "results/validation/gradient_check.passed"
    log: "logs/check_gradients.log"
    shell:
        """
        python tools/check_gradient_flow.py \
            --backbone_path {input.backbone} > {log} 2>&1
        touch {output}
        echo "✅ 梯度流验证通过" >> {log}
        """


rule clean:
    """清理中间文件"""
    shell:
        """
        rm -rf results/*/positives/*.sam
        rm -rf results/*/negatives/*.sam
        echo "✅ 已清理SAM文件"
        """


rule clean_all:
    """清理所有结果"""
    shell:
        """
        rm -rf results/ logs/
        echo "✅ 已清理所有结果"
        """


rule stats:
    """统计数据信息"""
    input:
        training_fasta="results/all_species_aggregated/e2e_training_data.fa",
        motif_coords="results/all_species_aggregated/all_motif_coordinates.tsv"
    output:
        "results/statistics/data_summary.txt"
    shell:
        """
        echo "数据统计信息" > {output}
        echo "============================================" >> {output}
        echo "" >> {output}
        
        echo "训练样本统计:" >> {output}
        echo "  总序列数: $(grep -c "^>" {input.training_fasta})" >> {output}
        echo "  SINE样本: $(grep -c "_SINE$" {input.training_fasta})" >> {output}
        echo "  nonSINE样本: $(grep -c "_nonSINE$" {input.training_fasta})" >> {output}
        echo "" >> {output}
        
        echo "Motif坐标统计:" >> {output}
        echo "  总记录数: $(tail -n +2 {input.motif_coords} | wc -l)" >> {output}
        echo "  完整检测(YES): $(tail -n +2 {input.motif_coords} | grep -c "YES" || echo 0)" >> {output}
        echo "" >> {output}
        
        echo "✅ 统计信息已保存"
        cat {output}
        """


# ============================================================================
# DAG可视化
# ============================================================================

rule dag:
    """生成workflow DAG图"""
    output:
        "workflow_dag.png"
    shell:
        """
        snakemake --dag | dot -Tpng > {output}
        echo "✅ Workflow DAG已保存到 {output}"
        """


rule rulegraph:
    """生成规则依赖图"""
    output:
        "workflow_rules.png"
    shell:
        """
        snakemake --rulegraph | dot -Tpng > {output}
        echo "✅ 规则依赖图已保存到 {output}"
        """