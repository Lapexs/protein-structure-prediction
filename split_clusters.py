#!/usr/bin/env python3
"""
Generate train/eval/test splits from RCSB sequence clusters for Team 2 (AFDB).
Supports custom dataset sizes (e.g. 50k: 40k train, 5k eval, 5k test).
"""

import argparse
import random
import sys
from pathlib import Path


def find_cluster_file(identity: int) -> Path:
    filename = f"clusters-by-entity-{identity}.txt"
    candidates = [
        Path(filename),
        Path(f"../{filename}"),
        Path(f"data/raw/{filename}"),
        Path(f"../data/raw/{filename}"),
        Path.home() / f"2_sem_DS/Bioinfa/{filename}"
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Could not find '{filename}' in any of {[str(c) for c in candidates]}")


def main():
    parser = argparse.ArgumentParser(description="Split protein clusters into train/eval/test sets.")
    parser.add_argument(
        "--identity", "-i",
        type=int,
        default=30,
        choices=[30, 50],
        help="Sequence identity threshold for clusters (30 or 50, default: 30)"
    )
    parser.add_argument(
        "--train",
        type=int,
        default=40000,
        help="Number of train proteins (default: 40000)"
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=5000,
        help="Number of eval proteins (default: 5000)"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=5000,
        help="Number of test proteins (default: 5000)"
    )
    parser.add_argument(
        "--members",
        type=int,
        default=10,
        help="Number of members to sample per cluster (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to save split files (default: data/splits)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for filenames (e.g. '_50k' -> train_af_50k.txt)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    
    cluster_file = find_cluster_file(args.identity)
    print(f"Reading clusters from {cluster_file} (identity: {args.identity}%)...")
    
    af_clusters = []
    with open(cluster_file, "r") as f:
        for line in f:
            # Consider only protein IDs prepended with AF_ (Team 2 requirement)
            members = [m for m in line.split() if m.startswith("AF_")]
            # Select clusters with at least required members
            if len(members) >= args.members:
                af_clusters.append(members)
                
    print(f"Found {len(af_clusters)} clusters with at least {args.members} AF_ members.")
    
    # Shuffle clusters (rows)
    random.shuffle(af_clusters)
    
    train_clusters_count = args.train // args.members
    eval_clusters_count = args.eval // args.members
    test_clusters_count = args.test // args.members
    total_needed = train_clusters_count + eval_clusters_count + test_clusters_count
    
    print(f"Clusters needed: {train_clusters_count} train + {eval_clusters_count} eval + {test_clusters_count} test = {total_needed} total.")
    
    if len(af_clusters) < total_needed:
        print(f"Error: Not enough clusters! Needed {total_needed}, found {len(af_clusters)}")
        sys.exit(1)
        
    selected_clusters = af_clusters[:total_needed]
    
    train_clusters = selected_clusters[:train_clusters_count]
    eval_clusters = selected_clusters[train_clusters_count:train_clusters_count + eval_clusters_count]
    test_clusters = selected_clusters[train_clusters_count + eval_clusters_count:]
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = args.suffix
    splits = [
        (f"train_af{suffix}.txt", train_clusters, args.train),
        (f"eval_af{suffix}.txt", eval_clusters, args.eval),
        (f"test_af{suffix}.txt", test_clusters, args.test)
    ]
    
    for filename, clusters, expected_count in splits:
        sampled_proteins = []
        for cluster in clusters:
            # For each cluster select random members
            sampled_members = random.sample(cluster, args.members)
            sampled_proteins.extend(sampled_members)
            
        assert len(sampled_proteins) == expected_count, f"Expected {expected_count}, got {len(sampled_proteins)}"
        
        target_path = out_dir / filename
        with open(target_path, "w") as out:
            for p_id in sampled_proteins:
                out.write(f"{p_id}\n")
        print(f"Saved {len(sampled_proteins)} proteins across {len(clusters)} clusters to {target_path}")

        # Also save in current directory if out_dir != '.'
        if str(out_dir) != ".":
            with open(filename, "w") as out:
                for p_id in sampled_proteins:
                    out.write(f"{p_id}\n")

    print("\n[Verification] Checking split overlap...")
    train_set = set(open(out_dir / f"train_af{suffix}.txt").read().splitlines())
    eval_set = set(open(out_dir / f"eval_af{suffix}.txt").read().splitlines())
    test_set = set(open(out_dir / f"test_af{suffix}.txt").read().splitlines())
    
    assert len(train_set & eval_set) == 0
    assert len(train_set & test_set) == 0
    assert len(eval_set & test_set) == 0
    print(f"SUCCESS: Zero overlap between splits! (Total proteins: {len(train_set) + len(eval_set) + len(test_set):,})")


if __name__ == "__main__":
    main()
