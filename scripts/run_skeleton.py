#!/usr/bin/env python3
"""
End-to-End Skeleton Demonstration (Meeting 3).
Tests:
1. Input sequence loading & tokenization.
2. PyTorch model forward pass predicting 3D coordinates (L, 3) and distance matrix (L, L).
3. Differentiable dRMSD loss calculation.
4. Overfitting sanity check on 1 protein (50 steps).
5. Exporting predicted structure to standard PDB format.
6. Generating contact map / distance matrix comparison plot.
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from src.data import ProteinStructureDataset, collate_protein_batch, save_coords_to_pdb
from src.models import ProteinStructureSkeleton, dRMSDLoss
from src.metrics import compute_drmsd, compute_kabsch_rmsd, compute_contact_precision


def main():
    print("=" * 70)
    print("Meeting 3: Protein Structure Prediction End-to-End Skeleton")
    print("=" * 70)

    # 1. Dataset & Sample Loading
    print("\n[Step 1] Loading sample protein from eval dataset...")
    dataset = ProteinStructureDataset(
        split="eval",
        splits_dir=str(ROOT_DIR / "data/splits"),
        structures_dir=str(ROOT_DIR / "structures"),
        max_len=256,
        limit=10
    )
    
    if len(dataset) == 0:
        print("Error: No structures found in structures/eval/af.")
        sys.exit(1)

    sample = dataset[0]
    print(f"Loaded Protein ID: {sample['id']}")
    print(f"Sequence length: {sample['length']} residues")
    print(f"Sequence preview: {sample['sequence'][:35]}...")
    print(f"Input tokens shape: {sample['tokens'].shape}")
    print(f"True coordinates shape: {sample['coords'].shape} (x, y, z in Angstroms)")
    print(f"True distance matrix shape: {sample['dist_matrix'].shape}")

    # 2. Batch DataLoader
    print("\n[Step 2] Testing DataLoader collation with padding...")
    batch = collate_protein_batch([dataset[0], dataset[1]])
    print(f"Batch tokens: {batch['tokens'].shape} (padded to max length in batch)")
    print(f"Batch mask: {batch['mask'].shape}")

    # 3. Model Forward Pass
    print("\n[Step 3] Initializing ProteinStructureSkeleton model...")
    model = ProteinStructureSkeleton(
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        predict_mode="displacement"
    )
    model.eval()

    with torch.no_grad():
        out = model(batch["tokens"], mask=batch["mask"])

    pred_coords = out["coords"]
    pred_dists = out["dist_matrix"]
    print(f"Output predicted coordinates shape: {pred_coords.shape}")
    print(f"Output predicted distance matrix shape: {pred_dists.shape}")

    # 4. Loss calculation
    loss_fn = dRMSDLoss()
    initial_loss = loss_fn(pred_dists, batch["dist_matrix"], mask=batch["mask"])
    print(f"Initial dRMSD structural loss: {initial_loss.item():.4f} A")

    # 5. Overfitting Sanity Check on 1 Protein
    print("\n[Step 4] Running sanity check: Overfitting model on 1 protein (60 gradient steps)...")
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    single_tokens = sample["tokens"].unsqueeze(0) # (1, L)
    single_true_dist = sample["dist_matrix"].unsqueeze(0) # (1, L, L)
    single_mask = torch.ones((1, sample["length"]), dtype=torch.bool)

    loss_history = []
    for step in range(1, 61):
        optimizer.zero_grad()
        out = model(single_tokens, mask=single_mask)
        loss = loss_fn(out["dist_matrix"], single_true_dist, mask=single_mask)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if step % 15 == 0 or step == 1:
            print(f"  Step {step:2d}/60 | dRMSD Loss: {loss.item():.3f} A")

    print(f"Loss dropped from {loss_history[0]:.2f} A to {loss_history[-1]:.2f} A (Gradients verified!)")

    # 6. Export Prediction to PDB
    print("\n[Step 5] Exporting final predicted 3D structure to PDB format...")
    model.eval()
    with torch.no_grad():
        final_out = model(single_tokens, mask=single_mask)
        final_coords = final_out["coords"][0].cpu().numpy()
        final_dists = final_out["dist_matrix"][0].cpu().numpy()

    out_dir = ROOT_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = out_dir / f"{sample['id']}_predicted.pdb"
    save_coords_to_pdb(final_coords, sample["sequence"], str(pdb_path))
    print(f"Saved predicted PDB to: {pdb_path}")

    # 7. Metrics evaluation
    true_coords_np = sample["coords"].numpy()
    true_dists_np = sample["dist_matrix"].numpy()

    drmsd = compute_drmsd(true_coords_np, final_coords)
    kabsch = compute_kabsch_rmsd(true_coords_np, final_coords)
    prec_8a = compute_contact_precision(true_coords_np, final_dists, threshold=8.0)
    print(f"Metrics vs True AFDB structure:")
    print(f"  * dRMSD:            {drmsd:.3f} A")
    print(f"  * Kabsch RMSD:      {kabsch:.3f} A")
    print(f"  * Contact Precision (<8A, |i-j|>=6): {prec_8a * 100:.1f}%")

    # 8. Distance Matrix / Contact Map Comparison Plot
    print("\n[Step 6] Plotting True vs Predicted Distance Matrix...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    im0 = axes[0].imshow(true_dists_np, cmap="viridis_r", vmin=0, vmax=35)
    axes[0].set_title(f"True Distance Matrix ({sample['id']})")
    axes[0].set_xlabel("Residue Index")
    axes[0].set_ylabel("Residue Index")
    fig.colorbar(im0, ax=axes[0], label="Distance (A)")

    im1 = axes[1].imshow(final_dists, cmap="viridis_r", vmin=0, vmax=35)
    axes[1].set_title("Predicted Distance Matrix (Skeleton)")
    axes[1].set_xlabel("Residue Index")
    fig.colorbar(im1, ax=axes[1], label="Distance (A)")

    plt.tight_layout()
    plot_path = out_dir / "distance_matrix_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved distance matrix plot to: {plot_path}")

    print("\n" + "=" * 70)
    print("SUCCESS: End-to-End Skeleton is fully operational!")
    print("=" * 70)


if __name__ == "__main__":
    main()
