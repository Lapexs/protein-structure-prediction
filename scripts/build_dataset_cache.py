#!/usr/bin/env python3
"""
Pre-parses CIF structures into compact binary PyTorch tensor caches (.pt).
Eliminates disk I/O bottlenecks and mmCIF parsing overhead during model training.
Can handle 50,000 proteins in ~150-200MB instead of ~15GB raw CIF files.
"""

import argparse
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data import load_split, af_id_to_filename, parse_cif_ca


def process_single_structure(args_tuple) -> Optional[Dict[str, Any]]:
    pid, cif_path, max_len = args_tuple
    try:
        seq, coords = parse_cif_ca(str(cif_path))
        if not seq or len(coords) == 0:
            return None
        if len(seq) != len(coords):
            return None
            
        if max_len and len(seq) > max_len:
            seq = seq[:max_len]
            coords = coords[:max_len]

        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        return {
            "id": pid,
            "sequence": seq,
            "coords": coords_tensor,
            "length": len(seq)
        }
    except Exception:
        return None


def build_cache_for_split(
    split: str,
    splits_dir: Path,
    structures_dir: Path,
    out_dir: Path,
    workers: int = 8,
    max_len: Optional[int] = None,
    limit: Optional[int] = None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split}.pt"

    protein_ids = load_split(split, splits_dir=str(splits_dir))
    if limit:
        protein_ids = protein_ids[:limit]

    split_struct_dir = structures_dir / split / "af"
    all_struct_dirs = [
        split_struct_dir,
        structures_dir / "train" / "af",
        structures_dir / "eval" / "af",
        structures_dir / "test" / "af",
        structures_dir,
    ]

    print(f"\n[Split: {split}] Resolving CIF files for {len(protein_ids)} protein IDs...")
    tasks = []
    missing_count = 0

    for pid in protein_ids:
        fname = af_id_to_filename(pid)
        if not fname:
            missing_count += 1
            continue

        target_path = None
        for s_dir in all_struct_dirs:
            p = s_dir / fname
            if p.exists() and p.stat().st_size > 0:
                target_path = p
                break

        if target_path:
            tasks.append((pid, target_path, max_len))
        else:
            missing_count += 1

    print(f"Found {len(tasks)} CIF files on disk ({missing_count} not yet downloaded / missing).")
    if not tasks:
        print(f"No structures found for split '{split}'. Skipping cache generation.")
        return

    print(f"Parsing structures with {workers} parallel processes...")
    cached_data = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_single_structure, task) for task in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Caching {split}", unit="prot"):
            res = f.result()
            if res is not None:
                cached_data.append(res)

    print(f"Saving binary tensor cache ({len(cached_data)} items) to {out_file}...")
    torch.save({
        "split": split,
        "count": len(cached_data),
        "data": cached_data
    }, out_file)

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    avg_len = sum(d["length"] for d in cached_data) / max(len(cached_data), 1)
    print(f"✓ Cache generated successfully: {out_file} ({file_size_mb:.2f} MB, avg length: {avg_len:.1f} residues).")


def main():
    parser = argparse.ArgumentParser(description="Pre-parse CIF files into PyTorch binary tensor caches.")
    parser.add_argument("--split", choices=["eval", "test", "train", "all"], default="all", help="Split to cache")
    parser.add_argument("--splits-dir", default=str(ROOT_DIR / "data/splits"), help="Path to splits directory")
    parser.add_argument("--structures-dir", default=str(ROOT_DIR / "structures"), help="Path to raw structures directory")
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "data/processed"), help="Output directory for .pt caches")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes")
    parser.add_argument("--max-len", type=int, default=None, help="Optional sequence truncation limit")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of proteins (for quick testing)")
    args = parser.parse_args()

    splits = ["eval", "test", "train"] if args.split == "all" else [args.split]

    splits_dir = Path(args.splits_dir)
    structures_dir = Path(args.structures_dir)
    out_dir = Path(args.out_dir)

    for sp in splits:
        build_cache_for_split(
            split=sp,
            splits_dir=splits_dir,
            structures_dir=structures_dir,
            out_dir=out_dir,
            workers=args.workers,
            max_len=args.max_len,
            limit=args.limit
        )


if __name__ == "__main__":
    main()
