#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_direct_predict_from_csv_no_label.py

修复版预测脚本（完全去除 label 依赖）

关键修改：
- process_csv_to_samples：不再读取或添加 'label'（预测不需要）
- 为兼容 SINEDatasetE2E，手动为每个 sample 添加 'label': 0（负样本模式）
  - 这样 _create_token_labels 会全打 O（不影响推理）
  - collate_fn 正常返回 label（但预测代码不使用）
- offset_mapping 处理：如果 dataset 未返回（原版 SINEDatasetE2E 没有），fallback 到“如果有前景信号，返回整个序列”（保守策略）
  - 建议你后续在 data.py 的 SINEDatasetE2E __getitem__ 中添加返回 offset_mapping（见文末注释）
- 其他保持不变：高阈值 + 强后处理 + 大上下文

预测时完全不需要 label 列（CSV 中可以有或没有，都无影响）
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig, TaskType
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from model import MotifGuidedSINEClassifier
from data import SINEDatasetE2E, collate_fn

# ================= 配置 =================
DEFAULT_MAX_TOKEN_LENGTH = 256
DEFAULT_MAX_BASE_LENGTH = 1600
BATCH_SIZE = 1
PROB_THRESHOLD = 0.90
MIN_REFINED_LEN = 50

# ================= 后处理工具 =================
import scipy.ndimage as ndimage

def extract_refined_sequence(pred_tags, offset_mapping, raw_seq, min_len=MIN_REFINED_LEN):
    """改进版后处理"""
    pred_tags = np.array(pred_tags)
    foreground = pred_tags > 0
    
    # 填充小断裂
    foreground = ndimage.binary_dilation(foreground, iterations=3)
    
    # 取最大连通组件
    labeled, num_features = ndimage.label(foreground)
    if num_features == 0:
        return "NO_SIGNAL", -1, -1
    
    sizes = np.bincount(labeled.ravel())[1:]
    largest_label = np.argmax(sizes) + 1
    best_mask = (labeled == largest_label)
    
    indices = np.where(best_mask)[0]
    start_idx, end_idx = indices[0], indices[-1] + 1
    
    # 如果有 offset_mapping，用精确 char 位置
    if offset_mapping is not None and len(offset_mapping) > 0:
        char_start = int(offset_mapping[start_idx][0])
        char_end = int(offset_mapping[end_idx][1])
    else:
        # fallback：按 token 比例估算（不精确，但比全序列好）
        total_tokens = len(pred_tags)
        char_start = int((start_idx / total_tokens) * len(raw_seq))
        char_end = int((end_idx / total_tokens) * len(raw_seq))
    
    char_start = max(0, char_start)
    char_end = min(len(raw_seq), char_end)
    
    refined_seq = raw_seq[char_start:char_end]
    if len(refined_seq) < min_len:
        return "TOO_SHORT", -1, -1
    
    return refined_seq, char_start, char_end

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--mask_pt", required=True)
    parser.add_argument("--motif_pos_tsv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--backbone", default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")
    parser.add_argument("--output_dir", default="./predictions_direct")
    parser.add_argument("--max_token_length", type=int, default=DEFAULT_MAX_TOKEN_LENGTH)
    parser.add_argument("--max_base_length", type=int, default=DEFAULT_MAX_BASE_LENGTH)
    parser.add_argument("--threshold", type=float, default=PROB_THRESHOLD)
    parser.add_argument("--min_len", type=int, default=MIN_REFINED_LEN)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_tsv = Path(args.output_dir) / "predictions.tsv"
    out_fasta = Path(args.output_dir) / "predictions_refined.fasta"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device(  "cpu")
    print(f"Using device: {device}")

    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    backbone = AutoModelForMaskedLM.from_pretrained(args.backbone, trust_remote_code=True)
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"]
    )
    backbone = get_peft_model(backbone, peft_config)

    model = MotifGuidedSINEClassifier(
        backbone=backbone,
        hidden_dim=256,
        num_classes=2,
        num_token_labels=4,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    # state_dict = checkpoint['model_state'] if isinstance(checkpoint, dict) and 'model_state' in checkpoint else checkpoint
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint

    # cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}

    cleaned = {}
    for k, v in state_dict.items():
        # 只去掉 DDP 的 module. 前缀
        new_k = k.replace('module.', '')
        
        # 修复 confidence 模块的命名不一致 (根据你上次的报错)
        if 'confidence_net' in new_k:
            new_k = new_k.replace('confidence_net', 'confidence_module.net')
        
        # 注意：不要 replace('base_model.model.')！！
        # 因为在 MotifGuidedSINEClassifier 结构中，
        # backbone 成员变量本身就是个 PeftModel，它需要这个层级。
        
        cleaned[new_k] = v

    # 3. 加载并检查
    msg = model.load_state_dict(cleaned, strict=False)
    print(f"✅ Loading Result: {msg}")
    
    # cleaned = {k.replace('module.', '').replace('base_model.model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print("✅ Model loaded")

    # 2. 处理 CSV → samples（不读取 label）
    def process_csv_to_samples(csv_path):
        df = pd.read_csv(csv_path)
        samples = []
        for _, row in df.iterrows():
            chrom = row['chrom']
            s, e = row['start'], row['end']
            strand = row['strand']
            uid = f"{chrom}:{s}-{e}({strand})"
            
            if strand == '-':
                rc = lambda x: str(Seq(x).reverse_complement()) if pd.notna(x) else ""
                full_seq = rc(row.get('flank_right', '')) + rc(row.get('seq', '')) + rc(row.get('flank_left', ''))
                core_start = len(rc(row.get('flank_right', '')))
                core_end = core_start + len(rc(row.get('seq', '')))
            else:
                full_seq = (row.get('flank_left', '') or '') + (row.get('seq', '') or '') + (row.get('flank_right', '') or '')
                core_start = len(row.get('flank_left', '') or '')
                core_end = core_start + len(row.get('seq', '') or '')
            
            samples.append({
                'uid': uid,
                'seq': full_seq,
                'core_start': core_start,
                'core_end': core_end,
                'label': 0  # 强制为 0（预测时无影响，仅为兼容 dataset）
            })
        return samples

    samples = process_csv_to_samples(args.input_csv)

    # 3. Dataset & DataLoader
    dataset = SINEDatasetE2E(
        samples=samples,
        mask_path=args.mask_pt,
        tokenizer=tokenizer,
        max_token_length=args.max_token_length,
        is_training=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 4. 推理
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            motif_mask = batch['motif_mask'].to(device)
            uids = batch['unique_ids']

            global_logits, emissions, _ = model(input_ids, att_mask, motif_mask)
            probs = torch.softmax(global_logits, dim=1)
            # print(probs)
            sine_probs = probs[:, 1].cpu().numpy()
            # print(sine_probs)
            preds = (sine_probs >= args.threshold).astype(int)

            raw_model = model.module if hasattr(model, 'module') else model
            decoded_tags_list = raw_model.decode(emissions, att_mask)

            for i, uid in enumerate(uids):
                refined_seq = "NON_SINE"
                if preds[i] == 1:
                    if preds[i] == 1:
                        # 1. 拿到 Dataset 计算出的精确 offset
                        # batch['offset_mapping'] 现在是 [Batch, SeqLen, 2]
                        offset = batch['offset_mapping'][i].cpu().numpy()
                        
                        # 2. 拿到本次 Tokenization 对应的原始序列文本
                        # 防止你 CSV 里的 sequence 和 Dataset 处理后的不一致（如裁剪过）
                        raw_seq = batch['raw_sequence'][i]
                        
                        # 3. 进行精准切片
                        refined_seq, char_start, char_end = extract_refined_sequence(
                            decoded_tags_list[i], 
                            offset, 
                            raw_seq
                        )
                results.append({
                    'unique_id': uid,
                    'SINE_prob': f"{sine_probs[i]:.4f}",
                    'Prediction': 'SINE' if preds[i] == 1 else 'Non-SINE',
                    'Refined_Sequence': refined_seq
                })

    # 5. 输出
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_tsv, sep='\t', index=False)
    print(f"✅ Predictions saved to {out_tsv}")

    records = []
    invalid = {"NO_SIGNAL", "NON_SINE", "TOO_SHORT", ""}
    for _, row in res_df.iterrows():
        seq = row['Refined_Sequence']
        if row['Prediction'] == 'SINE' and seq not in invalid and len(seq) >= args.min_len:
            records.append(SeqRecord(Seq(seq), id=f"{row['unique_id']}|prob={row['SINE_prob']}"))
    
    if records:
        SeqIO.write(records, out_fasta, "fasta")
        print(f"✅ {len(records)} refined sequences saved to {out_fasta}")

if __name__ == "__main__":
    main()