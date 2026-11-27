#!/usr/bin/env python3
"""
CLI script for Meeting 4: Protein Structure Prediction Evaluation Framework.

Compares predicted CA-only PDB models against ground-truth reference structures
(AlphaFold DB .cif or PDB .pdb) using:
  - Zhang Group TM-score  -> RMSD, TM_score, GDT_TS
  - OpenStructure (ost)   -> CA-only lDDT (--bb-lddt)

Saves results to a CSV file with exact columns:
    proteinID,RMSD,TM_score,GDT_TS,lDDT
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import evaluate_predictions_to_csv


def resolve_pairs(
    pred_dir: Path,
    ref_dir: Path,
    pred_suffix: str = "",
) -> List[Tuple[str, Path, Path]]:
    """Match predicted .pdb files in pred_dir to reference .cif/.pdb files in ref_dir."""
    pairs = []
    for pred_path in sorted(pred_dir.glob("*.pdb")):
        stem = pred_path.stem
        protein_id = stem[: -len(pred_suffix)] if pred_suffix and stem.endswith(pred_suffix) else stem

        # Look for matching reference in .cif or .pdb format
        candidates = [
            ref_dir / f"{protein_id}.cif",
            ref_dir / f"{protein_id}.pdb",
            ref_dir / f"{protein_id}-model_v6.cif",
        ]
        ref_path = next((c for c in candidates if c.exists()), None)
        if ref_path is None:
            # Recursive search fallback inside ref_dir
            matches = list(ref_dir.glob(f"**/{protein_id}.cif")) + list(ref_dir.glob(f"**/{protein_id}.pdb"))
            ref_path = matches[0] if matches else None

        if ref_path is not None:
            pairs.append((protein_id, pred_path, ref_path))
        else:
            print(f"[WARNING] No reference structure found for {protein_id} in {ref_dir}, skipping.")

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted protein structures using TM-score and OpenStructure (CA-only lDDT)."
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing predicted .pdb files",
    )
    parser.add_argument(
        "--ref-dir",
        type=str,
        default=str(ROOT_DIR / "structures/eval/af"),
        help="Directory containing ground-truth reference structures (.cif or .pdb)",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT_DIR / "results/evaluation_results.csv"),
        help="Output CSV path (columns: proteinID,RMSD,TM_score,GDT_TS,lDDT)",
    )
    parser.add_argument(
        "--pred-suffix",
        type=str,
        default="",
        help="Optional suffix to strip from prediction filenames when matching references (e.g. '_esmfold_ca')",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    ref_dir = Path(args.ref_dir)
    out_csv = Path(args.out_csv)

    pairs = resolve_pairs(pred_dir, ref_dir, pred_suffix=args.pred_suffix)
    if not pairs:
        print(f"Error: No matching prediction/reference pairs found between {pred_dir} and {ref_dir}.")
        sys.exit(1)

    print(f"Found {len(pairs)} prediction/reference pairs. Running TM-score & OpenStructure (CA-only lDDT)...")
    df = evaluate_predictions_to_csv(pairs, out_csv=out_csv)

    print("\n=== Evaluation Results ===")
    print(df.to_string(index=False))
    print("\n=== Summary (Mean) ===")
    print(df[["RMSD", "TM_score", "GDT_TS", "lDDT"]].mean().round(4).to_string())
    print(f"\nSaved CSV to: {out_csv}")


if __name__ == "__main__":
    main()
