#!/usr.bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
========

Contains common utility functions used across the SINE finder pipeline,
such as command execution, sequence manipulation, and interval operations.
"""
import subprocess
from pathlib import Path
from typing import List, Tuple

from Bio.Seq import Seq

# -------------------------------------------------
# Process & File Utilities
# -------------------------------------------------

def run_command(cmd: List[str], **kwargs):
    """
    Executes a shell command, prints it, and checks for errors.

    Args:
        cmd (List[str]): The command and its arguments as a list of strings.
        **kwargs: Additional keyword arguments to pass to subprocess.run().
    """
    print("[RUN]", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_minimap2_index(genome_path: str) -> str:
    """
    Checks if a minimap2 index (.mmi) exists for a genome, creating it if not.

    Args:
        genome_path (str): Path to the genome FASTA file.

    Returns:
        str: Path to the minimap2 index file.
    """
    index_path = genome_path + ".mmi"
    if not Path(index_path).exists():
        print(f"[INFO] minimap2 index not found. Creating {index_path}...")
        run_command(["minimap2", "-d", index_path, genome_path])
    return index_path

# -------------------------------------------------
# Sequence Utilities
# -------------------------------------------------

def revcomp(s: str) -> str:
    """
    Returns the reverse complement of a DNA sequence.

    Args:
        s (str): The input DNA sequence.

    Returns:
        str: The reverse complemented sequence.
    """
    return str(Seq(s).reverse_complement())

# -------------------------------------------------
# GFF/Interval Utilities
# -------------------------------------------------

Interval = Tuple[int, int]

def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """
    Merges a list of overlapping or adjacent intervals.

    Args:
        intervals (List[Interval]): A list of (start, end) tuples.

    Returns:
        List[Interval]: A list of merged intervals.
    """
    if not intervals:
        return []
    
    # Sort intervals by their start position
    intervals.sort()
    
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        
        if current_start <= prev_end:
            # Overlap or adjacent, merge them
            merged[-1] = (prev_start, max(prev_end, current_end))
        else:
            # No overlap, add the new interval
            merged.append((current_start, current_end))
            
    return merged


def invert_intervals(intervals: List[Interval], chrom_len: int) -> List[Interval]:
    """
    Given a list of "occupied" intervals, returns the "gaps" (e.g., intergenic regions).

    Args:
        intervals (List[Interval]): A sorted list of merged, occupied intervals.
        chrom_len (int): The total length of the chromosome.

    Returns:
        List[Interval]: A list of intervals representing the gaps.
    """
    if not intervals:
        return [(0, chrom_len)]
    
    gaps = []
    prev_end = 0
    
    for start, end in intervals:
        if start > prev_end:
            gaps.append((prev_end, start))
        prev_end = end
        
    if prev_end < chrom_len:
        gaps.append((prev_end, chrom_len))
        
    return gaps