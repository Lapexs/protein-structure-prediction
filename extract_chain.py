#!/usr/bin/env python3
# Usage:
#   cat train_pdb.txt | python extract_chain.py structures/train/pdb chains/train/

import sys
from pathlib import Path
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import MMCIFParser, MMCIFIO, Select

entry_id, in_dir, out_dir = sys.argv[1:4]
pdb_id, entity_id = entry_id.rsplit("_", 1)

d = MMCIF2Dict(f"{in_dir}/{pdb_id}.cif")
chains = {c for c, e in zip(d["_struct_asym.id"], d["_struct_asym.entity_id"]) if e == entity_id}

class Keep(Select):
    def accept_chain(self, chain): return chain.id in chains

st = MMCIFParser(QUIET=True).get_structure(pdb_id, f"{in_dir}/{pdb_id}.cif")
io = MMCIFIO()
io.set_structure(st)
Path(out_dir).mkdir(parents=True, exist_ok=True)
io.save(f"{out_dir}/{entry_id}.cif", Keep())
