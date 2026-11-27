"""
Unit tests for Meeting 4: Evaluation Framework (TM-score + OpenStructure CA-only lDDT).
"""

import csv
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import (
    find_tmscore_binary,
    find_ost_binary,
    evaluate_structure_pair,
    evaluate_predictions_to_csv,
)


def test_binaries_exist():
    tm_bin = find_tmscore_binary(ROOT_DIR)
    ost_bin = find_ost_binary(ROOT_DIR)
    assert tm_bin.exists(), f"TMscore missing: {tm_bin}"
    assert ost_bin.exists(), f"OpenStructure ost missing: {ost_bin}"


def test_identity_sanity_check():
    pid = "AF-A0A075B695-F1-model_v6"
    gt_ca_pdb = ROOT_DIR / "predictions/meeting4_gt_ca_test" / f"{pid}.pdb"
    ref_cif = ROOT_DIR / "structures/eval/af" / f"{pid}.cif"

    res = evaluate_structure_pair(pid, gt_ca_pdb, ref_cif)
    assert res["RMSD"] < 0.01, f"Expected RMSD ~ 0.0, got {res['RMSD']}"
    assert res["TM_score"] > 0.999, f"Expected TM_score ~ 1.0, got {res['TM_score']}"
    assert res["GDT_TS"] > 0.999, f"Expected GDT_TS ~ 1.0, got {res['GDT_TS']}"
    assert res["lDDT"] > 0.999, f"Expected lDDT ~ 1.0, got {res['lDDT']}"


def test_random_prediction_metrics():
    pid = "AF-A0A075B695-F1-model_v6"
    dummy_pdb = ROOT_DIR / "predictions/meeting4_dummy_test" / f"{pid}.pdb"
    ref_cif = ROOT_DIR / "structures/eval/af" / f"{pid}.cif"

    res = evaluate_structure_pair(pid, dummy_pdb, ref_cif)
    assert res["RMSD"] > 10.0, f"Expected high RMSD for random prediction, got {res['RMSD']}"
    assert res["TM_score"] < 0.25, f"Expected low TM_score for random prediction, got {res['TM_score']}"
    assert res["GDT_TS"] < 0.25, f"Expected low GDT_TS for random prediction, got {res['GDT_TS']}"
    assert res["lDDT"] < 0.25, f"Expected low lDDT for random prediction, got {res['lDDT']}"


def test_real_esmfold_prediction_and_csv_format():
    pid = "AF-A0A075B695-F1-model_v6"
    real_pdb = ROOT_DIR / "predictions/meeting4_real_test" / f"{pid}_esmfold_ca.pdb"
    ref_cif = ROOT_DIR / "structures/eval/af" / f"{pid}.cif"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_csv = Path(tmp_dir) / "eval.csv"
        df = evaluate_predictions_to_csv(
            [(pid, real_pdb, ref_cif)],
            out_csv=out_csv,
            show_progress=False,
        )
        assert list(df.columns) == ["proteinID", "RMSD", "TM_score", "GDT_TS", "lDDT"]
        assert len(df) == 1
        row = df.iloc[0]
        assert row["TM_score"] > 0.5, f"Expected high TM_score for real prediction, got {row['TM_score']}"
        assert row["GDT_TS"] > 0.5, f"Expected high GDT_TS for real prediction, got {row['GDT_TS']}"
        assert row["lDDT"] > 0.5, f"Expected high lDDT for real prediction, got {row['lDDT']}"

        # Verify exact CSV header string
        with open(out_csv) as f:
            header = f.readline().strip()
        assert header == "proteinID,RMSD,TM_score,GDT_TS,lDDT", f"Unexpected header: {header}"


if __name__ == "__main__":
    test_binaries_exist()
    test_identity_sanity_check()
    test_random_prediction_metrics()
    test_real_esmfold_prediction_and_csv_format()
    print("All Meeting 4 evaluation framework tests passed!")
