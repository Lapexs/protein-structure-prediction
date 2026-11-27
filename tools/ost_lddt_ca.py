#!/usr/bin/env python3
"""
OpenStructure (ost) helper script for CA-only lDDT evaluation.

Usage:
    ost tools/ost_lddt_ca.py <predicted_model.pdb|cif> <reference_structure.cif|pdb>

Prints a JSON object to stdout: {"lDDT": <float>, "ref_ca_count": <int>, "model_ca_count": <int>}
"""

import json
import sys
from pathlib import Path

from ost import io
from ost.mol.alg import lddt


def load_structure(path_str: str):
    path = Path(path_str)
    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return io.LoadMMCIF(str(path))
    return io.LoadPDB(str(path))


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: ost tools/ost_lddt_ca.py <model_path> <reference_path>\n")
        sys.exit(2)

    model_path = sys.argv[1]
    ref_path = sys.argv[2]

    ref_ent = load_structure(ref_path)
    mdl_ent = load_structure(model_path)

    # Select CA atoms only for CA-only lDDT evaluation
    ref_ca = ref_ent.Select("aname=CA")
    mdl_ca = mdl_ent.Select("aname=CA")

    if ref_ca.GetAtomCount() == 0:
        raise ValueError(f"No CA atoms found in reference structure: {ref_path}")
    if mdl_ca.GetAtomCount() == 0:
        raise ValueError(f"No CA atoms found in model structure: {model_path}")

    # Align chain naming if model and reference have single chains with differing IDs
    ref_chains = [c.GetName() for c in ref_ca.chains]
    mdl_chains = [c.GetName() for c in mdl_ca.chains]
    chain_mapping = None
    if len(ref_chains) == 1 and len(mdl_chains) == 1 and ref_chains[0] != mdl_chains[0]:
        chain_mapping = {mdl_chains[0]: ref_chains[0]}

    # Compute official OpenStructure CA-only (backbone-only) lDDT
    scorer = lddt.lDDTScorer(ref_ca, bb_only=True)
    score, _ = scorer.lDDT(
        mdl_ca,
        chain_mapping=chain_mapping,
        check_resnames=False,
    )

    result = {
        "lDDT": float(score),
        "ref_ca_count": int(ref_ca.GetAtomCount()),
        "model_ca_count": int(mdl_ca.GetAtomCount()),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
