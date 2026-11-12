#!/usr/bin/env python3
import argparse
import random
import sys
from pathlib import Path

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
        "--suffix",
        type=str,
        default="",
        help="Optional suffix for output filenames (e.g. '_30' -> train_af_30.txt)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    
    cluster_file = f"clusters-by-entity-{args.identity}.txt"
    if not Path(cluster_file).exists():
        print(f"Error: {cluster_file} not found. Please download it first.")
        sys.exit(1)
        
    print(f"Reading clusters from {cluster_file} (identity: {args.identity}%)...")
    
    af_clusters = []
    with open(cluster_file, "r") as f:
        for line in f:
            # Consider only protein IDs prepended with AF_ (Team 2 requirement)
            members = [m for m in line.split() if m.startswith("AF_")]
            # Select clusters with at least 10 members
            if len(members) >= 10:
                af_clusters.append(members)
                
    print(f"Found {len(af_clusters)} clusters with at least 10 AF_ members.")
    
    # Shuffle clusters (rows)
    random.shuffle(af_clusters)
    
    # Check if we have enough clusters
    # 4000 train (400 clusters) + 500 eval (50 clusters) + 500 test (50 clusters) = 500 clusters
    train_clusters_count = 400
    eval_clusters_count = 50
    test_clusters_count = 50
    total_needed = train_clusters_count + eval_clusters_count + test_clusters_count
    
    if len(af_clusters) < total_needed:
        print(f"Error: Not enough clusters! Needed {total_needed}, found {len(af_clusters)}")
        sys.exit(1)
        
    selected_clusters = af_clusters[:total_needed]
    
    train_clusters = selected_clusters[:train_clusters_count]
    eval_clusters = selected_clusters[train_clusters_count:train_clusters_count + eval_clusters_count]
    test_clusters = selected_clusters[train_clusters_count + eval_clusters_count:]
    
    suffix = args.suffix
    splits = [
        (f"train_af{suffix}.txt", train_clusters, 4000),
        (f"eval_af{suffix}.txt", eval_clusters, 500),
        (f"test_af{suffix}.txt", test_clusters, 500)
    ]
    
    for filename, clusters, expected_count in splits:
        sampled_proteins = []
        for cluster in clusters:
            # For each cluster select 10 random members
            sampled_members = random.sample(cluster, 10)
            sampled_proteins.extend(sampled_members)
            
        assert len(sampled_proteins) == expected_count, f"Expected {expected_count}, got {len(sampled_proteins)}"
        
        with open(filename, "w") as out:
            for p_id in sampled_proteins:
                out.write(f"{p_id}\n")
        print(f"Saved {len(sampled_proteins)} proteins across {len(clusters)} clusters to {filename}")

if __name__ == "__main__":
    main()
