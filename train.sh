# 1. 定义你想使用的显卡 ID
gpus="0,1"

# 2. 自动计算显卡数量 (通过统计逗号数量 + 1)
num_gpus=$(echo $gpus | tr "," "\n" | wc -l)

echo "使用 GPU: $gpus, 进程数: $num_gpus"

database_dir="/home/xuminghua/sine_classifier"

# 3. 运行 torchrun
CUDA_VISIBLE_DEVICES=$gpus \
OMP_NUM_THREADS=16 \
MKL_NUM_THREADS=16 \
OPENBLAS_NUM_THREADS=16 \
NUMEXPR_NUM_THREADS=16 \
VECLIB_MAXIMUM_THREADS=16 \
torchrun --master_port 29508 --nproc_per_node=$num_gpus train_e2e_classifier_crf2.py  \
 --backbone_path $database_dir/NT_finetuned_500M \
 --train_csv $database_dir/dataset_v1/train_aggressive_cleaned.csv \
 --train_motif_tsv $database_dir/dataset_v1/train_aggressive_cleaned_motif_pos2.tsv \
 --train_mask $database_dir/dataset_v1/train_aggressive_cleaned_masks2.pt \
 --val_csv $database_dir/dataset_v1/test_aggressive_cleaned.csv \
 --val_motif_tsv $database_dir/dataset_v1/test_val_aggressive_cleaned_motif_pos2.tsv \
 --val_mask $database_dir/dataset_v1/test_val_aggressive_cleaned_masks2.pt \
 --output_dir "./checkpoints_v1" \
 --batch_size 22 \
 --freeze_epochs 1 \
 --max_length 100 \
 --dropout 0.3
 
 