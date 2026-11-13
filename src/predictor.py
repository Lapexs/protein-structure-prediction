"""
End-to-End Protein Structure Prediction Skeleton Pipeline (Meeting 3).

Key components:
1. Multi-protein FASTA parsing using scikit-bio (`skbio.io.read()`).
2. Input sequence encoding into tensor of shape (B, D, N) with vocabulary size N=22:
   - 20 canonical amino acids
   - X for unknown/non-standard residues
   - PAD for padding
3. Mask tensor (B, D) identifying valid residues vs padding positions.
4. Modular dummy PyTorch model accepting (B, D, N) and mask (B, D), returning (L, 3) CA coordinates.
5. Export to PDB format strictly following wwPDB Format v3.3 ATOM record specification for CA atoms only.
   (Writes exactly one record per processed residue; does not write records for padded positions).
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import skbio.io

# ---------------------------------------------------------------------------
# 1. Vocabulary & Amino Acid Mapping (N = 22)
# ---------------------------------------------------------------------------
CANONICAL_AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
]
# N = 22 (20 standard + X + PAD)
VOCAB = CANONICAL_AMINO_ACIDS + ['X', 'PAD']
AA_TO_IDX = {aa: i for i, aa in enumerate(VOCAB)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

VOCAB_SIZE = len(VOCAB)  # 22
PAD_IDX = AA_TO_IDX['PAD']  # 21
UNK_IDX = AA_TO_IDX['X']    # 20

ONE_TO_THREE = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
    'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
    'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    'X': 'UNK', 'PAD': 'PAD'
}


# ---------------------------------------------------------------------------
# 2. FASTA Parsing with scikit-bio
# ---------------------------------------------------------------------------
def parse_fasta_skbio(fasta_path: Union[str, Path]) -> List[Tuple[str, str, str]]:
    """
    Parses a multi-protein FASTA file using scikit-bio (`skbio.io.read()`).
    
    Args:
        fasta_path: Path to the FASTA file.
        
    Returns:
        List of tuples: (protein_id, description, sequence)
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    records = []
    # skbio.io.read returns an iterator over Sequence objects for multi-sequence FASTA
    for seq_obj in skbio.io.read(str(fasta_path), format="fasta"):
        # Extract ID and optional description from sequence metadata
        meta = getattr(seq_obj, "metadata", {}) or {}
        pid = meta.get("id", "PROT")
        desc = meta.get("description", "")
        raw_seq = str(seq_obj).strip().upper()
        records.append((pid, desc, raw_seq))

    return records


# ---------------------------------------------------------------------------
# 3. Sequence Encoding & Masking (B, D, N)
# ---------------------------------------------------------------------------
def encode_sequence_one_hot(
    sequence: str,
    max_len: int = 1024,
    vocab_size: int = VOCAB_SIZE
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Encodes a single amino acid sequence into one-hot format:
    - Shorter sequences are padded with PAD token.
    - Longer sequences are truncated to max_len.
    - Mask identifies valid residues (True) versus padding (False).
    
    Args:
        sequence: 1-letter amino acid sequence.
        max_len: Maximum length D (default: 1024).
        vocab_size: Vocabulary size N (default: 22).
        
    Returns:
        one_hot: Tensor of shape (D, N)
        mask: Boolean Tensor of shape (D,)
        L: min(len(sequence), max_len)
    """
    L_orig = len(sequence)
    L = min(L_orig, max_len)
    
    one_hot = torch.zeros((max_len, vocab_size), dtype=torch.float32)
    mask = torch.zeros(max_len, dtype=torch.bool)
    
    for i in range(L):
        aa = sequence[i]
        idx = AA_TO_IDX.get(aa, UNK_IDX)
        one_hot[i, idx] = 1.0
        mask[i] = True
        
    for i in range(L, max_len):
        one_hot[i, PAD_IDX] = 1.0
        mask[i] = False
        
    return one_hot, mask, L


def batch_encode_sequences(
    sequences: List[str],
    max_len: int = 1024
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Collate and encode a list of sequences into batch tensors.
    
    Args:
        sequences: List of amino acid sequences.
        max_len: Sequence length D (default: 1024).
        
    Returns:
        x: Tensor of shape (B, D, N) where B=batch_size, D=max_len, N=22.
        mask: Boolean Tensor of shape (B, D)
        lengths: List of actual processed lengths [L_1, ..., L_B]
    """
    batch_x = []
    batch_mask = []
    lengths = []
    
    for seq in sequences:
        oh, m, L = encode_sequence_one_hot(seq, max_len=max_len)
        batch_x.append(oh)
        batch_mask.append(m)
        lengths.append(L)
        
    x = torch.stack(batch_x, dim=0)       # (B, D, N)
    mask = torch.stack(batch_mask, dim=0) # (B, D)
    return x, mask, lengths


# ---------------------------------------------------------------------------
# 4. Dummy PyTorch Predictor (Modular Template)
# ---------------------------------------------------------------------------
class DummyStructurePredictor(nn.Module):
    """
    Modular PyTorch predictor template for Protein Structure Prediction.
    
    Specifications:
    - Input shape: (B, D, N), where:
        B: Batch size (e.g. 16)
        D: Maximum sequence length (e.g. 1024)
        N: Amino-acid vocabulary size (22)
    - Mask shape: (B, D), identifying valid residues vs padding positions.
    - Output shape (per protein): (L, 3) where L = min(original_length, D).
    
    Note: Keep implementation modular and customizable. Once a real architecture
    is integrated (Transformer, BiLSTM, ESMFold head), it can replace this class
    without modifying the data pipeline or PDB export logic.
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_len: int = 1024,
        coord_dim: int = 3,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.coord_dim = coord_dim
        
        # Lightweight linear projection so the dummy model is a valid nn.Module with parameters
        self.proj = nn.Linear(vocab_size, coord_dim, bias=True)
        
        if seed is not None:
            torch.manual_seed(seed)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass accepting encoded sequences.
        
        Args:
            x: Tensor of shape (B, D, N)
            mask: Optional boolean Tensor of shape (B, D)
            
        Returns:
            coords: Tensor of shape (B, D, 3) with predicted CA coordinates.
        """
        B, D, N = x.shape
        # Generate random coordinates (scaled to standard molecular dimensions)
        # combined with a small projection of x for gradient/parameter validation
        random_coords = torch.randn(B, D, self.coord_dim, device=x.device, dtype=x.dtype) * 15.0
        coords = random_coords + 0.01 * self.proj(x)
        
        # Zero out coordinates for padded positions if mask is provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1).to(coords.dtype)
            
        return coords

    def predict_coordinates(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lengths: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        """
        Inference method returning per-protein coordinate tensors of shape (L_i, 3).
        Truncates/slices coordinates to exactly min(original_length, D).
        Padded positions are never returned.
        
        Returns:
            List of Tensors of shape (L_i, 3).
        """
        self.eval()
        with torch.no_grad():
            coords_batch = self.forward(x, mask=mask)  # (B, D, 3)
            
        results = []
        B = coords_batch.shape[0]
        for b in range(B):
            if lengths is not None:
                L = lengths[b]
            elif mask is not None:
                L = int(mask[b].sum().item())
            else:
                L = coords_batch.shape[1]
                
            # Slices exactly the valid L residues: (L, 3)
            coords_protein = coords_batch[b, :L, :].detach().cpu()
            results.append(coords_protein)
            
        return results


# ---------------------------------------------------------------------------
# 5. wwPDB Format v3.3 ATOM Record Generator (CA only)
# ---------------------------------------------------------------------------
def format_ca_atom_record(
    serial: int,
    res_name: str,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    occupancy: float = 1.00,
    temp_factor: float = 0.00
) -> str:
    """
    Formats exactly one ATOM record for a CA atom following wwPDB Format v3.3 specification.
    
    Columns definition (wwPDB):
      1 -  6: Record name   "ATOM  "
      7 - 11: Integer       serial (Atom serial number)
     13 - 16: Atom          name   (" CA ")
          17: Character     altLoc (" ")
     18 - 20: Residue name  resName (3-letter code, e.g. "ALA")
          22: Character     chainID (Chain identifier, e.g. "A")
     23 - 26: Integer       resSeq (Residue sequence number)
          27: AChar         iCode  (" ")
     31 - 38: Real(8.3)     x (X coordinate in Angstroms)
     39 - 46: Real(8.3)     y (Y coordinate in Angstroms)
     47 - 54: Real(8.3)     z (Z coordinate in Angstroms)
     55 - 60: Real(6.2)     occupancy (Occupancy, default 1.00)
     61 - 66: Real(6.2)     tempFactor (Temperature factor, default 0.00)
     77 - 78: LString(2)    element (Element symbol, right-justified, " C")
     79 - 80: LString(2)    charge ("  ")
     
    Returns:
        Formatted string strictly 80 characters wide ending with newline.
    """
    if len(res_name) == 1:
        res_3 = ONE_TO_THREE.get(res_name.upper(), "UNK")[:3]
    elif len(res_name) == 3:
        res_3 = res_name.upper()
    else:
        res_3 = res_name.upper()[:3]
    chain = chain_id[:1] if chain_id else "A"
    
    line = (
        f"ATOM  {serial:5d}  CA  {res_3:3s} {chain:1s}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          "
        f" C  \n"
    )
    return line


def write_ca_pdb(
    coords: Union[torch.Tensor, np.ndarray],
    sequence: str,
    out_path: Union[str, Path],
    protein_id: str = "PROT",
    chain_id: str = "A"
) -> Path:
    """
    Writes a PDB file containing ATOM records for CA atoms only, strictly
    following the wwPDB ATOM record specification.
    
    Rules:
    - Writes exactly one CA ATOM record per processed residue.
    - Does NOT write records for padded positions.
    - Length L = min(len(sequence), len(coords)).
    
    Args:
        coords: Tensor or ndarray of shape (L, 3).
        sequence: Amino acid sequence string.
        out_path: Output destination path.
        protein_id: Identifier of the protein.
        chain_id: PDB chain identifier (default "A").
        
    Returns:
        Path to the saved PDB file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(coords, torch.Tensor):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = np.asarray(coords)
        
    L = min(len(coords_np), len(sequence))
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"HEADER    PREDICTED STRUCTURE (CA SKELETON) - {protein_id}\n")
        f.write(f"TITLE     PREDICTED CA-ONLY MODEL WITH {L} RESIDUES\n")
        f.write("REMARK    GENERATED BY TEAM 2 (PSP) END-TO-END SKELETON PIPELINE\n")
        
        for i in range(L):
            serial = i + 1
            res_seq = i + 1
            aa = sequence[i]
            x, y, z = coords_np[i]
            
            record = format_ca_atom_record(
                serial=serial,
                res_name=aa,
                chain_id=chain_id,
                res_seq=res_seq,
                x=float(x),
                y=float(y),
                z=float(z),
                occupancy=1.00,
                temp_factor=0.00
            )
            f.write(record)
            
        last_aa = sequence[L - 1] if L > 0 else "X"
        last_res3 = ONE_TO_THREE.get(last_aa, "UNK")[:3]
        f.write(f"TER   {L + 1:5d}      {last_res3:3s} {chain_id:1s}{L:4d}\n")
        f.write("END\n")
        
    return out_path


# ---------------------------------------------------------------------------
# 6. Complete End-to-End Pipeline Runner
# ---------------------------------------------------------------------------
def predict_fasta_to_pdb(
    fasta_path: Union[str, Path],
    out_dir: Union[str, Path],
    model: Optional[nn.Module] = None,
    batch_size: int = 16,
    max_len: int = 1024,
    device: str = "cpu"
) -> List[Path]:
    """
    Executes the complete end-to-end predictor pipeline:
    1. Reads multi-protein FASTA using scikit-bio (`skbio.io.read`).
    2. Batches sequences into shape (B, D, N) with mask (B, D).
    3. Runs PyTorch model inference.
    4. Slices valid coordinates to shape (L, 3) where L = min(original_length, D).
    5. Exports PDB files containing only CA ATOM records (no padding records).
    
    Args:
        fasta_path: Path to multi-protein FASTA.
        out_dir: Directory where PDB files will be saved.
        model: PyTorch model (defaults to DummyStructurePredictor).
        batch_size: Inference batch size B (e.g. 16).
        max_len: Max sequence length D (e.g. 1024).
        device: 'cpu', 'cuda', or 'mps'.
        
    Returns:
        List of generated PDB file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse FASTA with scikit-bio
    records = parse_fasta_skbio(fasta_path)
    if not records:
        raise ValueError(f"No valid sequences found in {fasta_path}")
        
    # 2. Initialize model if not provided
    if model is None:
        model = DummyStructurePredictor(vocab_size=VOCAB_SIZE, max_len=max_len)
    model.to(device)
    model.eval()
    
    generated_pdbs = []
    
    # 3. Process in batches
    for start_idx in range(0, len(records), batch_size):
        batch_records = records[start_idx : start_idx + batch_size]
        batch_pids = [r[0] for r in batch_records]
        batch_seqs = [r[2] for r in batch_records]
        
        # Encode batch -> (B, D, N) and mask (B, D)
        x, mask, lengths = batch_encode_sequences(batch_seqs, max_len=max_len)
        x = x.to(device)
        mask = mask.to(device)
        
        # Model inference -> List of (L_i, 3)
        if hasattr(model, "predict_coordinates"):
            pred_coords_list = model.predict_coordinates(x, mask=mask, lengths=lengths)
        else:
            with torch.no_grad():
                out = model(x, mask=mask)
                pred_coords_list = [out[b, : lengths[b], :].cpu() for b in range(len(lengths))]
                
        # 4. Write PDB files for each protein in batch
        for pid, seq, coords, L in zip(batch_pids, batch_seqs, pred_coords_list, lengths):
            # Clean filename
            safe_pid = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in pid)
            pdb_path = out_dir / f"{safe_pid}.pdb"
            
            write_ca_pdb(
                coords=coords,
                sequence=seq,
                out_path=pdb_path,
                protein_id=pid,
                chain_id="A"
            )
            generated_pdbs.append(pdb_path)
            
    return generated_pdbs
