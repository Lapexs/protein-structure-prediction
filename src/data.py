"""Data loading, ID translation, and structure parsing utilities for AFDB."""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

def af_id_to_url(protein_id: str) -> Optional[str]:
    """Convert an RCSB entity ID (e.g. AF_AFQ5V1K0F1_1) to an AlphaFold EBI CIF download URL."""
    m = re.match(r"AF_AF(.+?)(F\d+)_\d+$", protein_id.strip())
    if not m:
        return None
    uniprot_id = m.group(1)
    fragment_id = m.group(2)
    return f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-{fragment_id}-model_v6.cif"

def af_id_to_filename(protein_id: str) -> Optional[str]:
    """Convert an RCSB entity ID to its corresponding CIF filename."""
    m = re.match(r"AF_AF(.+?)(F\d+)_\d+$", protein_id.strip())
    if not m:
        return None
    return f"AF-{m.group(1)}-{m.group(2)}-model_v6.cif"

def load_split(split: str, splits_dir: str = "data/splits", suffix: str = "") -> List[str]:
    """Load protein IDs for a given split ('train', 'eval', 'test')."""
    candidates = [
        Path(splits_dir) / f"{split}_af{suffix}.txt",
        Path(f"{split}_af{suffix}.txt"),
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r") as f:
                return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError(f"Could not find split file for '{split}' in {[str(c) for c in candidates]}")

def parse_cif_ca(cif_path: str) -> Tuple[str, List[Tuple[float, float, float]]]:
    """
    Fast native CIF parser extracting amino acid sequence and C-alpha (CA) coordinates.
    Returns:
        sequence (str): 1-letter amino acid sequence
        coordinates (list of (x, y, z)): 3D coordinates of CA atoms
    """
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    sequence = []
    coords = []
    
    in_atom_site = False
    col_names = []
    
    with open(cif_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("loop_"):
                in_atom_site = False
                col_names = []
                continue
                
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_names.append(line.split()[0])
                continue
                
            if in_atom_site:
                if line.startswith("#") or line.startswith("_"):
                    in_atom_site = False
                    continue
                    
                parts = line.split()
                if len(parts) >= len(col_names):
                    record = dict(zip(col_names, parts[:len(col_names)]))
                    atom_name = record.get("_atom_site.auth_atom_id") or record.get("_atom_site.label_atom_id")
                    
                    if atom_name == "CA":
                        res_name = record.get("_atom_site.auth_comp_id") or record.get("_atom_site.label_comp_id")
                        x = float(record.get("_atom_site.Cartn_x", 0.0))
                        y = float(record.get("_atom_site.Cartn_y", 0.0))
                        z = float(record.get("_atom_site.Cartn_z", 0.0))
                        
                        aa = three_to_one.get(res_name, 'X')
                        sequence.append(aa)
                        coords.append((x, y, z))
                        
    return "".join(sequence), coords
