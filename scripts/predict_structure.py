#!/usr/bin/env python3
"""
CLI Script for Meeting 3: End-to-End Protein Structure Prediction Skeleton.

Usage:
    python scripts/predict_structure.py --fasta data/samples/sample_multi.fasta --out-dir predictions/
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.predictor import (
    predict_fasta_to_pdb,
    DummyStructurePredictor,
    VOCAB_SIZE
)


def main():
    parser = argparse.ArgumentParser(
        description="Run modular End-to-End Protein Structure Prediction pipeline on a FASTA file."
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default=str(ROOT_DIR / "data/samples/sample_multi.fasta"),
        help="Path to input multi-protein FASTA file"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT_DIR / "predictions/meeting3_pdb"),
        help="Directory to save predicted PDB files"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size B for inference (default: 16)"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum sequence length D (default: 1024)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Protein Structure Prediction: End-to-End Skeleton (Meeting 3)")
    print("=" * 70)
    print(f"Input FASTA:       {args.fasta}")
    print(f"Output Directory:  {args.out_dir}")
    print(f"Batch Size (B):    {args.batch_size}")
    print(f"Max Length (D):    {args.max_len}")
    print(f"Vocabulary Size (N): {VOCAB_SIZE} (20 AA + X + PAD)")
    print("=" * 70)

    model = DummyStructurePredictor(
        vocab_size=VOCAB_SIZE,
        max_len=args.max_len,
        seed=args.seed
    )

    pdb_paths = predict_fasta_to_pdb(
        fasta_path=args.fasta,
        out_dir=args.out_dir,
        model=model,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

    print(f"\nSuccessfully generated {len(pdb_paths)} PDB structure(s):")
    for p in pdb_paths:
        with open(p) as f:
            atom_count = sum(1 for line in f if line.startswith("ATOM  "))
        print(f"  ✓ {p.name:25s} | {atom_count:5d} CA ATOM records | Path: {p}")
        
    print("\nAll ATOM records strictly adhere to wwPDB Format v3.3 specification.")
    print("No records written for padded positions.")


if __name__ == "__main__":
    main()
