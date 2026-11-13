"""
Unit tests for Meeting 3 End-to-End Skeleton Pipeline.
"""

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch

from src.predictor import (
    parse_fasta_skbio,
    encode_sequence_one_hot,
    batch_encode_sequences,
    DummyStructurePredictor,
    format_ca_atom_record,
    write_ca_pdb,
    predict_fasta_to_pdb,
    VOCAB_SIZE,
    PAD_IDX,
    UNK_IDX,
    AA_TO_IDX
)


def test_vocabulary_size():
    assert VOCAB_SIZE == 22
    assert PAD_IDX == 21
    assert UNK_IDX == 20
    assert AA_TO_IDX['A'] == 0


def test_encode_and_masking():
    seq = "ACDEF"  # len = 5
    max_len = 10
    one_hot, mask, L = encode_sequence_one_hot(seq, max_len=max_len)
    
    assert one_hot.shape == (10, 22)
    assert mask.shape == (10,)
    assert L == 5
    
    # Valid residues
    assert mask[:5].all()
    assert (one_hot[:5, :20].sum(dim=-1) == 1.0).all()
    
    # Padded residues
    assert (~mask[5:]).all()
    assert (one_hot[5:, PAD_IDX] == 1.0).all()


def test_truncation():
    seq = "A" * 1500
    max_len = 1024
    one_hot, mask, L = encode_sequence_one_hot(seq, max_len=max_len)
    assert L == 1024
    assert mask.all()
    assert one_hot.shape == (1024, 22)


def test_batch_encoding():
    seqs = ["ACDEF", "WY" * 10, "M"]
    x, mask, lengths = batch_encode_sequences(seqs, max_len=64)
    assert x.shape == (3, 64, 22)
    assert mask.shape == (3, 64)
    assert lengths == [5, 20, 1]


def test_dummy_model_shapes():
    B, D, N = 4, 128, 22
    x = torch.randn(B, D, N)
    mask = torch.ones(B, D, dtype=torch.bool)
    mask[0, 50:] = False
    lengths = [50, 128, 128, 128]
    
    model = DummyStructurePredictor(vocab_size=N, max_len=D)
    coords_list = model.predict_coordinates(x, mask=mask, lengths=lengths)
    
    assert len(coords_list) == B
    assert coords_list[0].shape == (50, 3)
    assert coords_list[1].shape == (128, 3)


def test_wwpdb_atom_record_specification():
    record = format_ca_atom_record(
        serial=1,
        res_name="ALA",
        chain_id="A",
        res_seq=1,
        x=10.123,
        y=-4.567,
        z=0.890
    )
    # Exclude trailing newline
    line = record.rstrip("\r\n")
    assert len(line) == 80
    assert line[0:6] == "ATOM  "
    assert line[6:11] == "    1"
    assert line[12:16] == " CA "
    assert line[17:20] == "ALA"
    assert line[21] == "A"
    assert line[22:26] == "   1"
    assert line[30:38] == "  10.123"
    assert line[38:46] == "  -4.567"
    assert line[46:54] == "   0.890"
    assert line[54:60] == "  1.00"
    assert line[60:66] == "  0.00"
    assert line[76:78] == " C"


def test_no_records_for_padded_positions():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(">test1\nACDEFGHIKL\n>test2\nMK\n")
        
        out_paths = predict_fasta_to_pdb(fasta_file, tmp_path, max_len=100)
        assert len(out_paths) == 2
        
        # test1: 10 residues -> exactly 10 ATOM records
        with open(out_paths[0]) as f:
            atom_lines_1 = [l for l in f if l.startswith("ATOM  ")]
        assert len(atom_lines_1) == 10
        
        # test2: 2 residues -> exactly 2 ATOM records (NO padding records written)
        with open(out_paths[1]) as f:
            atom_lines_2 = [l for l in f if l.startswith("ATOM  ")]
        assert len(atom_lines_2) == 2


if __name__ == "__main__":
    test_vocabulary_size()
    test_encode_and_masking()
    test_truncation()
    test_batch_encoding()
    test_dummy_model_shapes()
    test_wwpdb_atom_record_specification()
    test_no_records_for_padded_positions()
    print("All tests passed successfully!")
