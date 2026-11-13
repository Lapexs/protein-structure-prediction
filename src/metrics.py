"""Evaluation metrics for protein structure prediction (Meeting 4)."""

import math
from typing import List, Tuple, Union
import numpy as np

def compute_drmsd(
    coords_true: Union[List[Tuple[float, float, float]], np.ndarray],
    coords_pred: Union[List[Tuple[float, float, float]], np.ndarray]
) -> float:
    """
    Compute distance-RMSD (dRMSD) between pairwise distance matrices.
    Rotation- and translation-invariant structural metric.
    """
    c_true = np.asarray(coords_true, dtype=np.float32)
    c_pred = np.asarray(coords_pred, dtype=np.float32)
    
    n = len(c_true)
    if n != len(c_pred) or n < 2:
        return 0.0
        
    diff_true = c_true[:, np.newaxis, :] - c_true[np.newaxis, :, :]
    d_true = np.sqrt(np.sum(diff_true ** 2, axis=-1))

    diff_pred = c_pred[:, np.newaxis, :] - c_pred[np.newaxis, :, :]
    d_pred = np.sqrt(np.sum(diff_pred ** 2, axis=-1))
    
    # Exclude diagonal
    triu_indices = np.triu_indices(n, k=1)
    diff = d_true[triu_indices] - d_pred[triu_indices]
    
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_kabsch_rmsd(
    coords_true: Union[List[Tuple[float, float, float]], np.ndarray],
    coords_pred: Union[List[Tuple[float, float, float]], np.ndarray]
) -> float:
    """
    Compute coordinate Root Mean Square Deviation (RMSD) after optimal Kabsch superposition.
    """
    P = np.asarray(coords_true, dtype=np.float64)
    Q = np.asarray(coords_pred, dtype=np.float64)
    
    if len(P) != len(Q) or len(P) == 0:
        return 0.0

    # Center coordinates at centroid
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    p_centered = P - centroid_P
    q_centered = Q - centroid_Q

    # Covariance matrix
    H = np.dot(p_centered.T, q_centered)
    U, S, Vt = np.linalg.svd(H)
    
    # Optimal rotation matrix
    R = np.dot(Vt.T, U.T)
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    q_aligned = np.dot(q_centered, R)
    diff = p_centered - q_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
    return float(rmsd)


def compute_contact_precision(
    coords_true: Union[List[Tuple[float, float, float]], np.ndarray],
    coords_pred: Union[List[Tuple[float, float, float]], np.ndarray],
    threshold: float = 8.0,
    min_seq_sep: int = 6
) -> float:
    """
    Compute precision of predicted contacts (C-alpha distance < threshold Angstroms)
    for non-local residue pairs (|i - j| >= min_seq_sep).
    """
    c_true = np.asarray(coords_true, dtype=np.float32)
    c_pred = np.asarray(coords_pred, dtype=np.float32)
    n = len(c_true)
    
    if n != len(c_pred) or n <= min_seq_sep:
        return 0.0
        
    diff_true = c_true[:, np.newaxis, :] - c_true[np.newaxis, :, :]
    d_true = np.sqrt(np.sum(diff_true ** 2, axis=-1))

    diff_pred = c_pred[:, np.newaxis, :] - c_pred[np.newaxis, :, :]
    d_pred = np.sqrt(np.sum(diff_pred ** 2, axis=-1))
    
    true_contacts = (d_true < threshold)
    pred_contacts = (d_pred < threshold)
    
    # Mask out local pairs
    sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    valid_mask = sep >= min_seq_sep
    
    pred_positives = pred_contacts & valid_mask
    if np.sum(pred_positives) == 0:
        return 0.0
        
    tp = np.sum(pred_positives & true_contacts)
    return float(tp / np.sum(pred_positives))
