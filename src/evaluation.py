"""
Meeting 4: Evaluation Framework for Protein Structure Prediction.

Integrates community-standard external evaluation tools:
1. Zhang Group TM-score (https://zhanggroup.org/TM-score/) -> RMSD, TM_score, GDT_TS
2. OpenStructure (https://openstructure.org/) -> CA-only lDDT (--bb-lddt)

Exports results in the standardized CSV format:
    proteinID,RMSD,TM_score,GDT_TS,lDDT
"""

import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent


def find_tmscore_binary(root_dir: Optional[Path] = None) -> Path:
    """Locate the compiled Zhang Group TM-score binary."""
    base = Path(root_dir) if root_dir else ROOT_DIR
    candidates = [
        base / "tools" / "TMscore",
        Path("tools/TMscore"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    on_path = shutil.which("TMscore")
    if on_path:
        return Path(on_path)

    raise FileNotFoundError(
        "TMscore binary not found. Run 'bash tools/setup_evaluation_tools.sh' "
        "or compile tools/TMscore.cpp with: g++ -O3 -ffast-math -o tools/TMscore tools/TMscore.cpp"
    )


def find_ost_binary(root_dir: Optional[Path] = None) -> Path:
    """Locate the OpenStructure (ost) binary."""
    base = Path(root_dir) if root_dir else ROOT_DIR
    candidates = [
        base / ".ost_env" / "bin" / "ost",
        Path(".ost_env/bin/ost"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    on_path = shutil.which("ost")
    if on_path:
        return Path(on_path)

    raise FileNotFoundError(
        "OpenStructure ('ost') binary not found. Run 'bash tools/setup_evaluation_tools.sh' "
        "or install OpenStructure via micromamba/conda: "
        "micromamba create -y -p .ost_env -c conda-forge -c bioconda python=3.11 openstructure"
    )


def run_tmscore(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    tmscore_bin: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """
    Run Zhang Group TM-score comparing a predicted structure against a reference structure.
    Supports both .pdb and .cif files directly.

    Returns:
        dict with keys: 'RMSD', 'TM_score', 'GDT_TS'
    """
    pred_path = Path(pred_path).resolve()
    ref_path = Path(ref_path).resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicted structure not found: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference structure not found: {ref_path}")

    binary = Path(tmscore_bin) if tmscore_bin else find_tmscore_binary()

    proc = subprocess.run(
        [str(binary), str(pred_path), str(ref_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = proc.stdout

    rmsd_match = re.search(r"RMSD of\s+the common residues=\s*([0-9.]+)", stdout)
    tm_match = re.search(r"TM-score\s*=\s*([0-9.]+)", stdout)
    gdt_match = re.search(r"GDT-TS-score=\s*([0-9.]+)", stdout)

    if not (rmsd_match and tm_match and gdt_match):
        raise RuntimeError(
            f"Failed to parse TM-score output for {pred_path.name} vs {ref_path.name}.\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{proc.stderr}"
        )

    return {
        "RMSD": float(rmsd_match.group(1)),
        "TM_score": float(tm_match.group(1)),
        "GDT_TS": float(gdt_match.group(1)),
    }


def run_openstructure_lddt_ca(
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    ost_bin: Optional[Union[str, Path]] = None,
) -> float:
    """
    Run OpenStructure (`ost tools/ost_lddt_ca.py`) to compute official CA-only lDDT
    using `ost.mol.alg.lddt.lDDTScorer(ref.Select("aname=CA"), bb_only=True)`.
    Supports both .pdb and .cif structures (including pure CA-only predicted PDBs).

    Returns:
        float: CA-only lDDT score in [0.0, 1.0]
    """
    pred_path = Path(pred_path).resolve()
    ref_path = Path(ref_path).resolve()
    if not pred_path.exists():
        raise FileNotFoundError(f"Predicted structure not found: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference structure not found: {ref_path}")

    binary = Path(ost_bin) if ost_bin else find_ost_binary()
    helper_script = ROOT_DIR / "tools" / "ost_lddt_ca.py"
    if not helper_script.exists():
        raise FileNotFoundError(f"OpenStructure helper script not found: {helper_script}")

    cmd = [str(binary), str(helper_script), str(pred_path), str(ref_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"OpenStructure lDDT evaluation failed for {pred_path.name} vs {ref_path.name}.\n"
            f"CMD: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Parse last JSON line printed by ost_lddt_ca.py
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            data = json.loads(line)
            if "lDDT" in data:
                return float(data["lDDT"])

    raise RuntimeError(
        f"Could not parse JSON output from OpenStructure helper.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


def evaluate_structure_pair(
    protein_id: str,
    pred_path: Union[str, Path],
    ref_path: Union[str, Path],
    tmscore_bin: Optional[Union[str, Path]] = None,
    ost_bin: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single predicted structure against its ground-truth reference structure
    using TM-score (RMSD, TM_score, GDT_TS) and OpenStructure (CA-only lDDT).
    """
    tm_metrics = run_tmscore(pred_path, ref_path, tmscore_bin=tmscore_bin)
    lddt_val = run_openstructure_lddt_ca(pred_path, ref_path, ost_bin=ost_bin)

    return {
        "proteinID": protein_id,
        "RMSD": round(tm_metrics["RMSD"], 4),
        "TM_score": round(tm_metrics["TM_score"], 4),
        "GDT_TS": round(tm_metrics["GDT_TS"], 4),
        "lDDT": round(lddt_val, 4),
    }


def evaluate_predictions_to_csv(
    pairs: List[Tuple[str, Union[str, Path], Union[str, Path]]],
    out_csv: Union[str, Path],
    tmscore_bin: Optional[Union[str, Path]] = None,
    ost_bin: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a list of (protein_id, pred_path, ref_path) tuples and save the results
    to a CSV file with exact columns:
        proteinID,RMSD,TM_score,GDT_TS,lDDT
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tm_bin = Path(tmscore_bin) if tmscore_bin else find_tmscore_binary()
    ost_exec = Path(ost_bin) if ost_bin else find_ost_binary()

    rows = []
    iterator = tqdm(pairs, desc="Evaluating structures", unit="prot") if show_progress else pairs
    for pid, pred_p, ref_p in iterator:
        res = evaluate_structure_pair(
            protein_id=pid,
            pred_path=pred_p,
            ref_path=ref_p,
            tmscore_bin=tm_bin,
            ost_bin=ost_exec,
        )
        rows.append(res)

    fieldnames = ["proteinID", "RMSD", "TM_score", "GDT_TS", "lDDT"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return pd.DataFrame(rows, columns=fieldnames)
