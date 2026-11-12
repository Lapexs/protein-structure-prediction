"""Evaluation metrics for protein structure prediction (Meeting 4)."""

import math
from typing import List, Tuple

def compute_drmsd(coords_true: List[Tuple[float, float, float]], coords_pred: List[Tuple[float, float, float]]) -> float:
    """
    Compute distance-RMSD (dRMSD) between pairwise distance matrices.
    Rotation- and translation-invariant metric.
    """
    n = len(coords_true)
    if n != len(coords_pred) or n < 2:
        return 0.0
        
    diff_sum = 0.0
    pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            d_true = math.dist(coords_true[i], coords_true[j])
            d_pred = math.dist(coords_pred[i], coords_pred[j])
            diff = d_true - d_pred
            diff_sum += diff * diff
            pairs += 1
            
    return math.sqrt(diff_sum / pairs) if pairs > 0 else 0.0

def compute_rmsd_kabsch(coords_true, coords_pred) -> float:
    """
    Placeholder for optimal coordinate superposition RMSD (Kabsch algorithm).
    Will be fully integrated with numpy / PyTorch in Meeting 4.
    """
    return compute_drmsd(coords_true, coords_pred)
