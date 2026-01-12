#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_e2e_classifier.py (OBIE 4-Class)
"""
import argparse
import logging
from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from Bio.Seq import Seq
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
sys.path.insert(0, str(Path(__file__).parent.parent))
from data import SINEDatasetE2E, collate_fn
from model import MotifGuidedSINEClassifier, FocalLoss
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

logger = logging.getLogger(__name__)

import random

def set_seed(seed=42):
    # 1. Python & Numpy
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果是多 GPU
    
    # 3. CUDNN 决定性设置 (会略微降低训练速度，但能保证完全一致)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. 确保在分布式环境下所有进程的基础随机起点一致
    # 这一步非常重要，尤其是在数据预处理和采样时

def seed_worker(worker_id):
    # 确保每个子进程的种子既是确定的，又在不同进程间不同
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def revcomp(s):
    if pd.isna(s) or s == "": return ""
    return str(Seq(s).reverse_complement())

def setup_logging(output_dir, rank):
    handlers = [logging.StreamHandler()]
    if rank == 0:
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(output_dir) / "training.log"
            handlers.append(logging.FileHandler(log_file, mode='w'))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers, force=True)

def setup_ddp():
    if "RANK" not in os.environ:
        os.environ["RANK"], os.environ["LOCAL_RANK"], os.environ["WORLD_SIZE"] = "0", "0", "1"
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", "12355"
    dist.init_process_group(backend='nccl')
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

def cleanup_ddp(): dist.destroy_process_group()
def set_backbone_freeze(model, freeze: bool):
    raw = model.module if hasattr(model, "module") else model
    for p in raw.backbone.parameters(): p.requires_grad = not freeze

def process_csv_to_samples(csv_path, rank):
    """
    读取 CSV 并计算 OBIE 边界
    """
    if rank == 0: logger.info(f"Processing data from: {csv_path}")
    df = pd.read_csv(csv_path)
    samples = []
    
    for _, row in df.iterrows():
        chrom = row['chrom']
        s, e = row['start'], row['end']
        strand = row['strand']
        label = int(row['label'])
        uid = f"{chrom}:{s}-{e}({strand})"
        
        # 提取各部分序列 (处理空值)
        fl = row['flank_left'] if pd.notna(row['flank_left']) else ""
        core = row['seq'] if pd.notna(row['seq']) else ""
        fr = row['flank_right'] if pd.notna(row['flank_right']) else ""
        
        if strand == '-':
            # 负链: 物理上的右侧翼变成了 5'端 (Flank Left)
            # 拼接顺序: RC(Right) + RC(Core) + RC(Left)
            fl_rc = revcomp(fr)
            core_rc = revcomp(core)
            fr_rc = revcomp(fl)
            
            full_seq = fl_rc + core_rc + fr_rc
            core_start = len(fl_rc)
            core_end = len(fl_rc) + len(core_rc)
        else:
            # 正链: Flank Left + Core + Flank Right
            full_seq = fl + core + fr
            core_start = len(fl)
            core_end = len(fl) + len(core)
            
        samples.append({
            'uid': uid,
            'seq': full_seq,
            'label': label,
            'core_start': core_start,
            'core_end': core_end
        })
    return samples

def calculate_iou(pred_flat, label_flat):
    # 0=O, 1=B, 2=I, 3=E.  Foreground = {1, 2, 3}
    valid_mask = (label_flat != -100)
    pred = pred_flat[valid_mask]
    label = label_flat[valid_mask]
    if len(label) == 0: return 0.0
    
    pred_bin = (pred > 0).long()
    label_bin = (label > 0).long()
    
    intersection = (pred_bin & label_bin).sum().float()
    union = (pred_bin | label_bin).sum().float()
    if union == 0: return 1.0
    return (intersection / union).item()

@torch.no_grad()
def evaluate(model, loader, criterion_cls, device, rank, world_size):
    model.eval()
    total_loss, total_iou, iou_count = 0.0, 0.0, 0
    all_g_probs, all_g_preds, all_g_labels = [], [], []

    # DDP处理：获取原始模型以调用 decode 方法
    raw_model = model.module if hasattr(model, "module") else model

    # 确保所有进程都知道 loader 是否为空
    local_count = torch.tensor([len(loader)], device=device)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
    if local_count.item() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(loader, desc="Evaluate", disable=(rank != 0)):
        input_ids = batch['input_ids'].to(device)
        att_mask = batch['attention_mask'].to(device)
        motif_mask = batch['motif_mask'].to(device)
        labels = batch['label'].to(device)
        token_labels = batch['token_labels'].to(device) 

        # 传入 token_labels 以获取 CRF Loss
        g_logits, emissions, crf_loss = model(input_ids, att_mask, motif_mask, token_labels)

        # Loss 计算
        loss_cls = criterion_cls(g_logits, labels)
        # 权重建议：CRF Loss (NLL) 通常数值较大(10-100)，CLS Loss较小(0.1-1.0)
        # 建议给 CRF Loss 一个较小的系数，或者只是监控它
        loss = loss_cls + 0.5 * crf_loss

        total_loss += loss.item()

        # Global Metrics 收集
        probs = torch.softmax(g_logits, dim=1)
        preds = torch.argmax(g_logits, dim=1)
        all_g_probs.append(probs) # 保持在 GPU 上以便 gather，或者 cpu() 后再处理
        all_g_preds.append(preds)
        all_g_labels.append(labels)

        # Segmentation Metrics (使用 Viterbi 解码)
        # raw_model.decode 返回 List[List[int]]，每个样本长度可能不同（取决于mask）
        decoded_tags_list = raw_model.decode(emissions, att_mask)

        for i, pred_tags in enumerate(decoded_tags_list):
            # pred_tags: List[int], 长度为该样本的真实有效长度
            
            # 转换预测为 tensor
            p_tensor = torch.tensor(pred_tags, device=device)
            
            # 获取对应的真实标签 (需截取有效长度，忽略原本的 padding -100)
            valid_len = len(pred_tags)
            l_tensor = token_labels[i, :valid_len]
            
            # 计算 IoU
            iou = calculate_iou(p_tensor, l_tensor)
            total_iou += iou
            iou_count += 1

    # -----------------------------------------------------
    # DDP 数据收集逻辑 (从第一段代码集成并适配)
    # -----------------------------------------------------
    # 1. 拼接本地数据
    if len(all_g_probs) > 0:
        all_g_probs = torch.cat(all_g_probs) # Tensor on Device
        all_g_preds = torch.cat(all_g_preds)
        all_g_labels = torch.cat(all_g_labels)
    else:
        # 处理空数据情况
        all_g_probs = torch.empty((0, 2), device=device)
        all_g_preds = torch.empty((0,), dtype=torch.long, device=device)
        all_g_labels = torch.empty((0,), dtype=torch.long, device=device)

    # 2. 分布式环境下的 Reduce 和 Gather
    if world_size > 1:
        # 聚合标量指标 (Loss, IoU)
        metrics_tensor = torch.tensor([total_loss, total_iou, iou_count, len(loader)], device=device, dtype=torch.float64)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = metrics_tensor[0].item() / metrics_tensor[3].item()
        avg_iou = metrics_tensor[1].item() / metrics_tensor[2].item() if metrics_tensor[2].item() > 0 else 0.0

        # 聚合向量指标 (Probs, Preds, Labels)
        # 获取各卡数据大小
        local_size = torch.tensor([all_g_labels.size(0)], dtype=torch.long, device=device)
        size_list = [torch.tensor([0], dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [s.item() for s in size_list]
        max_size = max(sizes)
        
        # 动态获取类别数 (SINE 任务通常是 2 分类)
        num_classes_prob = all_g_probs.size(1) if all_g_probs.size(0) > 0 else 2
        
        # 准备 Buffer (带 Padding)
        prob_buffer = torch.zeros((max_size, num_classes_prob), dtype=all_g_probs.dtype, device=device)
        prob_buffer[:local_size] = all_g_probs
        gathered_probs = [torch.zeros_like(prob_buffer) for _ in range(world_size)]
        
        pred_buffer = torch.zeros((max_size,), dtype=all_g_preds.dtype, device=device)
        pred_buffer[:local_size] = all_g_preds
        gathered_preds = [torch.zeros_like(pred_buffer) for _ in range(world_size)]
        
        label_buffer = torch.zeros((max_size,), dtype=all_g_labels.dtype, device=device)
        label_buffer[:local_size] = all_g_labels
        gathered_labels = [torch.zeros_like(label_buffer) for _ in range(world_size)]

        # 执行 Gather
        dist.all_gather(gathered_probs, prob_buffer)
        dist.all_gather(gathered_preds, pred_buffer)
        dist.all_gather(gathered_labels, label_buffer)

        if rank == 0:
            final_probs = []
            final_preds = []
            final_labels = []
            # 去除 Padding
            for i, size in enumerate(sizes):
                final_probs.append(gathered_probs[i][:size].cpu())
                final_preds.append(gathered_preds[i][:size].cpu())
                final_labels.append(gathered_labels[i][:size].cpu())
            
            all_g_probs = torch.cat(final_probs).numpy()
            all_g_preds = torch.cat(final_preds).numpy()
            all_g_labels = torch.cat(final_labels).numpy()
        else:
            # 非 Rank 0 返回占位符
            return 0.0, 0.0, 0.0, 0.0, 0.0
    else:
        # 单机模式
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        avg_iou = total_iou / iou_count if iou_count > 0 else 0.0
        all_g_probs = all_g_probs.cpu().numpy()
        all_g_preds = all_g_preds.cpu().numpy()
        all_g_labels = all_g_labels.cpu().numpy()

    # 3. 计算最终指标 (仅 Rank 0)
    acc = accuracy_score(all_g_labels, all_g_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_g_labels, all_g_preds, average='weighted', zero_division=0
    )
    
    try:
        # 二分类取 index 1, 多分类通常需要 label_binarize 或 macro/weighted 处理
        if all_g_probs.shape[1] == 2:
            auc = roc_auc_score(all_g_labels, all_g_probs[:, 1])
        else:
            # 如果是多分类，这里简单置0或计算 Macro AUC
            auc = 0.0 
    except Exception:
        auc = 0.0

    return avg_loss, acc, f1, auc, avg_iou


def uniform_subsample(samples, target_size):
    """
    分层等间隔采样：保证正负样本比例不变，且覆盖整个数据集范围
    """
    if len(samples) <= target_size:
        return samples

    # 1. 按标签分组
    pos_samples = [s for s in samples if s['label'] == 1]
    neg_samples = [s for s in samples if s['label'] == 0]
    
    # 2. 计算比例
    pos_ratio = len(pos_samples) / len(samples)
    num_pos = max(1, int(target_size * pos_ratio)) # 至少取1个正样本
    num_neg = target_size - num_pos
    
    def get_indices(data_len, count):
        # 计算步长，实现等间隔采样
        if count <= 0: return []
        if count == 1: return [data_len // 2]
        return np.linspace(0, data_len - 1, count, dtype=int).tolist()

    # 3. 执行采样
    selected_pos = [pos_samples[i] for i in get_indices(len(pos_samples), num_pos)]
    selected_neg = [neg_samples[i] for i in get_indices(len(neg_samples), num_neg)]
    
    final_samples = selected_pos + selected_neg
    return final_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--train_mask", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--val_mask", required=True)
    parser.add_argument("--output_dir", required=True)
    
    # 仍需传入但实际上不怎么用的参数，为了兼容
    parser.add_argument("--train_motif_tsv", default="", help="Optional")
    parser.add_argument("--val_motif_tsv", default="", help="Optional")
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backbone_lr", type=float, default=5e-6)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=100)

    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    parser.add_argument("--debug", action="store_true", help="使用 1/1000 的数据快速跑通流程")

    # 【新增】LoRA 相关参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_lora", action="store_true", default=True, help="默认开启 LoRA")

    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()

    # 建议设置种子，可以设为参数 args.seed
    seed = 0
    set_seed(seed) 

    device = torch.device(f'cuda:{local_rank}')
    setup_logging(args.output_dir, rank)
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_path, trust_remote_code=True, use_fast=True)

    # 加载数据 (使用新函数)
    train_samples = process_csv_to_samples(args.train_csv, rank)
    val_samples = process_csv_to_samples(args.val_csv, rank)

    logger.info(f"max length \t {args.max_length}")

    # --- 新增快速验证逻辑 ---
    if args.debug:
        # 计算目标大小
        ratio = 0.001
        min_train = args.batch_size * world_size * 2
        min_val = args.batch_size * world_size * 1
        
        target_train_size = max(min_train, int(len(train_samples) * ratio))
        target_val_size = max(min_val, int(len(val_samples) * ratio))
        
        if rank == 0:
            logger.info(f"⚠️ DEBUG MODE: Performing Stratified Uniform Sampling (1/1000)...")
        
        train_samples = uniform_subsample(train_samples, target_train_size)
        val_samples = uniform_subsample(val_samples, target_val_size)
        
        if rank == 0:
            pos_in_debug = sum(1 for s in train_samples if s['label'] == 1)
            logger.info(f"Sampled Train: {len(train_samples)} (Pos: {pos_in_debug})")

    train_ds = SINEDatasetE2E(train_samples, args.train_mask, tokenizer, args.max_length, True)
    val_ds = SINEDatasetE2E(val_samples, args.val_mask, tokenizer, args.max_length, False)

    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone_path, trust_remote_code=True)

    # 2. 【关键修改】应用 LoRA
    if args.use_lora:
        if rank == 0: logger.info(f"🚀 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # 或者 None，取决于你的 Backbone 类型
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=0.1,
            # target_modules 需要根据你的模型结构调整。
            # 如果是 BERT/DNABERT，通常是 ["query", "key", "value"] 
            # 如果不确定，可以先打印 backbone 结构查看 attention 层的名字
            target_modules=["query", "key", "value", "dense"] 
        )
        backbone = get_peft_model(backbone, peft_config)
        
        if rank == 0:
            backbone.print_trainable_parameters()

    # 4分类: O, B, I, E
    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        num_token_labels=4, # OBIE
        dropout=args.dropout,
        freeze_backbone=False
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    
    # 不需要复杂的分组了，只提取 requires_grad=True 的参数
    # 这些参数包括：LoRA 的 adapter, 自定义的 Head, CRF 等
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # 也可以简单地区分一下 CRF 的学习率，如果需要的话
    crf_params = list(map(id, model.module.crf.parameters()))
    base_params = filter(lambda p: id(p) not in crf_params and p.requires_grad, model.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.head_lr},   # LoRA + Head 使用统一学习率
        {'params': model.module.crf.parameters(), 'lr': 1e-3} # CRF 保持较大
    ])

    if len(train_samples) > 0: 
        criterion_cls = FocalLoss(gamma=2.0, alpha=0.75).to(device)
    else: 
        criterion_cls = nn.CrossEntropyLoss().to(device)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_dl = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=4, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))
    val_dl = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=4, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(0))

    best_val_f1 = 0.0

    # --- 恢复训练逻辑 ---
    start_epoch = 1
    if args.resume:
        if rank == 0: logger.info(f"Resuming from checkpoint: {args.resume}")
        
        # 加载 checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        
        # 1. 加载模型权重
        # strict=False 很重要，万一你微调了模型结构，它能容忍非关键键值的差异
        # model.module 是因为你用了 DDP
        model.module.load_state_dict(checkpoint['model_state'], strict=False)
        
        # 2. 恢复 Epoch 计数
        start_epoch = checkpoint['epoch'] + 1
        
        # 3. [关键决策] 是否加载优化器状态？
        # 由于你修改了 Loss 逻辑，为了去除旧梯度的干扰，建议注释掉下面这两行，使用新的优化器重新开始
        # if 'optimizer' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     if rank == 0: logger.info("  ✅ Optimizer state loaded.")

        if rank == 0: logger.info(f"  ✅ Weights loaded. Resuming from Epoch {start_epoch}")

        # 4. 处理解冻状态
        # 如果恢复点已经过了冻结期，需要手动解冻 Backbone
        if start_epoch > args.freeze_epochs + 1:
            if rank == 0: logger.info(f"  🔓 Current epoch {start_epoch} > freeze_epochs {args.freeze_epochs}, Unfreezing backbone.")
            set_backbone_freeze(model, freeze=False)
        else:
            if rank == 0: logger.info(f"  🔒 Current epoch {start_epoch} <= freeze_epochs {args.freeze_epochs}, Backbone remains frozen.")
            set_backbone_freeze(model, freeze=True)

    # 定义 CRF Loss 的目标权重和预热轮数
    TARGET_CRF_WEIGHT = 0.05
    CRF_WARMUP_EPOCHS = 3  # 前5个epoch逐渐增加权重

    for epoch in range(start_epoch, args.epochs + 1):
        # --- [新增] 计算当前 Epoch 的 CRF 权重 ---
        if epoch <= CRF_WARMUP_EPOCHS:
            # 线性增长：第1轮=0.1, 第2轮=0.2... (假设目标0.5, warm=5)
            # 或者从0开始： (epoch - 1) / CRF_WARMUP_EPOCHS * TARGET
            current_crf_weight = (epoch / CRF_WARMUP_EPOCHS) * TARGET_CRF_WEIGHT
        else:
            current_crf_weight = TARGET_CRF_WEIGHT
            
        if rank == 0:
            logger.info(f"Epoch {epoch} | CRF Loss Weight: {current_crf_weight:.4f}")

        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}", disable=(rank!=0))
        
        for batch in pbar:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            m_mask = batch['motif_mask'].to(device)
            labels = batch['label'].to(device)
            t_labels = batch['token_labels'].to(device)

            # Forward: 传入 t_labels
            # g_out: 全局分类 Logits
            # emissions: CRF 发射分数 (用于 debug, 此时不用)
            # l_crf: CRF Negative Log Likelihood Loss
            g_out, emissions, l_crf = model(ids, mask, m_mask, t_labels, labels=labels)
            
            l_cls = criterion_cls(g_out, labels)

            # --- [修改] 使用动态权重计算总 Loss ---
            # 如果是 Warmup 初期，current_crf_weight 很小，模型主要优化 l_cls
            loss = l_cls + current_crf_weight * l_crf
            
            # 综合损失
            # l_crf 通常数值在 几十到几百，视 batch_size 和 seq_len 而定
            # l_cls 通常在 0.x
            # 建议系数: 0.1 或 0.05，防止 CRF loss 淹没分类 loss
            # 或者使用动态权重 (Multi-task weighting)，但在 DDP 下简单系数最稳
            # loss = l_cls + 0.01 * l_crf
            # [策略 B] 只对正样本 (label=1) 计算 CRF Loss
            # 这样避免模型在负样本上拼命学“全是背景”，导致特征坍塌
            # 注意：这需要改 model.py 返回 unreduced loss，或者在这里简单处理
            # 简单写法：如果全是负样本，就不加 l_crf
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if rank==0: 
                # [修改] 使用科学计数法，并处理 value 极小的情况
                # 记录一下当前的 crf loss 值，如果是 0 说明是负样本 batch
                crf_val = l_crf.item()
                cls_val = l_cls.item()
                
                # 只有当 crf_val > 0 时才认为是有意义的 CRF 训练
                # 这里的 formatting 让你看清到底是不是 0
                pbar.set_postfix({
                    'ls': f"{loss.item():.2e}", 
                    'cls': f"{cls_val:.2e}",
                    'crf': f"{crf_val:.2e}"
                })

        v_loss, v_acc, v_f1, v_auc, v_iou = evaluate(
            model, val_dl, criterion_cls, 
            device, rank, world_size
        )
        if rank == 0:
            logger.info(f"Epoch {epoch} | Val F1: {v_f1:.4f} | mIoU: {v_iou:.4f}")
            # 1. 保存每一轮的独立 Checkpoint
            logger.info(f" 正在写入模型")
            epoch_save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': v_f1,
                'val_iou': v_iou,
            }, epoch_save_path)
            logger.info(f"  💾 Saved epoch checkpoint to {epoch_save_path}")

            # 2. 原有的最佳模型保存逻辑
            if v_f1 > best_val_f1:
                best_val_f1 = v_f1
                torch.save(model.module.state_dict(), Path(args.output_dir)/"best_model.pt")
                logger.info(f"  ✅ New Best F1! Saved best_model.pt")

            # 3. 保存最新的状态 (用于意外中断恢复)
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_f1': best_val_f1
            }, Path(args.output_dir) / "latest.pt")

                
    cleanup_ddp()

if __name__ == "__main__":
    main()