"""Model architectures and baselines for protein structure prediction (Meeting 3 & 6)."""

import math
from typing import Tuple, Dict, Any, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProteinStructureSkeleton(nn.Module):
        """
        End-to-End trainable PyTorch model for Meeting 3 (PSP Skeleton).
        
        Architecture:
        1. Amino Acid Embedding: tokens (B, L) -> (B, L, embed_dim)
        2. Sequence Encoder: BiLSTM / 1D-Conv encoder (B, L, embed_dim) -> (B, L, hidden_dim * 2)
        3. Backbone Coordinate Head: predicts 3D coordinates (B, L, 3) with chain continuity.
        4. Pairwise Distance Matrix: computes D_ij = ||x_i - x_j||_2 for all residue pairs.
        """
        def __init__(
            self,
            vocab_size: int = 22,
            embed_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            pad_idx: int = 21,
            predict_mode: str = "displacement" # "displacement" or "direct"
        ):
            super().__init__()
            self.predict_mode = predict_mode
            self.pad_idx = pad_idx

            # 1. Residue Embedding
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

            # 2. Sequence Encoder (Bidirectional LSTM)
            self.encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if num_layers > 1 else 0.0
            )

            encoder_out_dim = hidden_dim * 2

            # 3. 3D Coordinate Projection Head
            self.coord_head = nn.Sequential(
                nn.Linear(encoder_out_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 3)
            )

        def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            """
            Args:
                tokens: (Batch, Length) LongTensor of amino acid indices
                mask: (Batch, Length) BoolTensor, True for valid residues, False for padding
            Returns:
                dict containing:
                  'coords': (Batch, Length, 3) predicted 3D C-alpha coordinates
                  'dist_matrix': (Batch, Length, Length) predicted pairwise distance matrix
            """
            B, L = tokens.shape

            # Embed sequence
            x = self.embedding(tokens) # (B, L, embed_dim)

            # Encode sequence
            feat, _ = self.encoder(x) # (B, L, 2 * hidden_dim)

            # Predict raw vectors
            raw_out = self.coord_head(feat) # (B, L, 3)

            if self.predict_mode == "displacement":
                # Chain continuity prior: predict consecutive step vectors normalized to ~3.8 Angstroms
                # (typical C-alpha to C-alpha distance in peptides)
                norm = torch.norm(raw_out, dim=-1, keepdim=True) + 1e-6
                step_vectors = raw_out / norm * 3.80
                
                # Start chain at origin (0,0,0) and cumulatively sum displacements
                coords = torch.cumsum(step_vectors, dim=1)
            else:
                coords = raw_out

            if mask is not None:
                coords = coords * mask.unsqueeze(-1)

            # Pairwise distance matrix: D_ij = ||x_i - x_j||_2
            # (B, L, 1, 3) - (B, 1, L, 3) -> (B, L, L, 3)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)
            dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)

            if mask is not None:
                pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
                dist_matrix = dist_matrix * pair_mask.float()

            return {
                "coords": coords,
                "dist_matrix": dist_matrix
            }


    class dRMSDLoss(nn.Module):
        """
        Differentiable distance-RMSD (dRMSD) loss.
        Rotation- and translation-invariant structural loss function.
        """
        def __init__(self, eps: float = 1e-6):
            super().__init__()
            self.eps = eps

        def forward(
            self,
            pred_dists: torch.Tensor,
            true_dists: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            Args:
                pred_dists: (B, L, L) predicted distance matrix
                true_dists: (B, L, L) ground truth distance matrix
                mask: (B, L) validity mask
            """
            diff_sq = (pred_dists - true_dists) ** 2

            if mask is not None:
                pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).float()
                # Zero out diagonal (distance of residue to itself is always 0)
                eye_mask = 1.0 - torch.eye(pred_dists.shape[1], device=pred_dists.device).unsqueeze(0)
                total_mask = pair_mask * eye_mask
                
                loss = torch.sum(diff_sq * total_mask) / (torch.sum(total_mask) + self.eps)
            else:
                loss = torch.mean(diff_sq)

            return torch.sqrt(loss + self.eps)

except ImportError:
    pass


class DummyStructurePredictor:
    """
    Minimal non-PyTorch baseline generating synthetic geometric helix.
    """
    def __init__(self):
        pass

    def predict(self, sequence: str):
        coords = []
        for i, _ in enumerate(sequence):
            angle = i * 1.745
            x = 2.5 * math.cos(angle)
            y = 2.5 * math.sin(angle)
            z = i * 1.5
            coords.append((x, y, z))
        return coords
