# 统计参考库中的低复杂度序列
from Bio import SeqIO
from collections import Counter

def check_repeat(seq_str, period=2, threshold=0.7):
    """检测是否是简单重复"""
    n = len(seq_str)
    for p in range(1, min(period + 1, n // 2)):
        matches = sum(1 for i in range(n - p) if seq_str[i] == seq_str[i+p])
        if matches / (n - p) > threshold:
            return True, p
    return False, 0

low_complexity_count = 0
total_count = 0

for record in SeqIO.parse("data/Dfam_RepBase/SINE.fa", "fasta"):
    total_count += 1
    seq = str(record.seq).upper()
    
    is_repeat, period = check_repeat(seq, period=10)
    if is_repeat:
        low_complexity_count += 1
        if low_complexity_count <= 10:  # 打印前 10 个
            print(f"Low-complexity: {record.id}")
            print(f"  Sequence: {seq[:100]}")
            print(f"  Period: {period}bp")
            print()

print(f"\nTotal: {total_count}")
print(f"Low-complexity: {low_complexity_count} ({100*low_complexity_count/total_count:.1f}%)")
