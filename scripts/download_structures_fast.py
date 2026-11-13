#!/usr/bin/env python3
"""
High-throughput multi-threaded downloader for AlphaFold DB CIF structures.
Features:
- Multi-threaded downloads with configurable concurrency (default: 8 workers).
- Skips already downloaded or cached structures (safe resume).
- Handles EBI HTTP 429 rate limiting with exponential backoff.
- Live progress reporting with tqdm.
"""

import argparse
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data import load_split, af_id_to_url, af_id_to_filename
from tqdm import tqdm


def download_single(url: str, dest_path: Path, max_retries: int = 5) -> Tuple[bool, str]:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, "skipped"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    retries = 0
    while retries < max_retries:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Bioinformatics-Student-PSP)"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                
            with open(temp_path, "wb") as f:
                f.write(data)
                
            temp_path.replace(dest_path)
            return True, "downloaded"

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_s = 5 * (2 ** retries)
                time.sleep(wait_s)
                retries += 1
            elif e.code == 404:
                return False, "404"
            else:
                time.sleep(2)
                retries += 1
        except Exception:
            time.sleep(2)
            retries += 1

    return False, "failed"


def main():
    parser = argparse.ArgumentParser(description="Download AlphaFold structures in parallel.")
    parser.add_argument("--split", choices=["eval", "test", "train", "all"], default="eval", help="Split to download (default: eval)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent workers (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    splits = ["eval", "test", "train"] if args.split == "all" else [args.split]

    for split in splits:
        protein_ids = load_split(split, splits_dir=str(ROOT_DIR / "data/splits"))
        if args.limit:
            protein_ids = protein_ids[:args.limit]

        out_dir = ROOT_DIR / "structures" / split / "af"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Split: {split}] Preparing to download {len(protein_ids)} structures to {out_dir}...")
        tasks = []
        for pid in protein_ids:
            url = af_id_to_url(pid)
            fname = af_id_to_filename(pid)
            if url and fname:
                tasks.append((url, out_dir / fname))

        downloaded = 0
        skipped = 0
        errors = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_url = {executor.submit(download_single, url, path): path for url, path in tasks}
            
            pbar = tqdm(as_completed(future_to_url), total=len(tasks), desc=f"Downloading {split}", unit="file")
            for future in pbar:
                success, status = future.result()
                if status == "skipped":
                    skipped += 1
                elif success:
                    downloaded += 1
                else:
                    errors += 1
                    
                pbar.set_postfix({"new": downloaded, "cached": skipped, "err": errors})

        print(f"Summary for {split}: {downloaded} new, {skipped} cached/existing, {errors} errors. Total available: {downloaded + skipped}/{len(tasks)}")


if __name__ == "__main__":
    main()
