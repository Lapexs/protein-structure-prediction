#!/usr/bin/env python3
"""Console-only sequence audit. Dependencies: biopython, NCBI BLAST+."""
"""
$env:Path = "D:\BLAST\blast-2.17.0+\bin;" + $env:Path
blastp -version  

New-Item -ItemType Directory -Force D:\psp_temp | Out-Null
$env:TEMP = "D:\psp_temp"
$env:TMP = "D:\psp_temp"
python audit_data.py


"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import itertools
import re
import shutil
import subprocess
import tempfile
import sys


def command(args):
    p = subprocess.run(list(map(str, args)), capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr.strip())
    return p.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--suffix', default='_50')
    parser.add_argument('--identity', type=float, default=50)
    parser.add_argument('--coverage', type=float, default=80)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not (0 < args.identity <= 100 and 0 < args.coverage <= 100 and args.threads > 0):
        parser.error('Invalid identity, coverage or threads')
    try:
        from Bio.PDB import MMCIFParser
        from Bio.SeqUtils import seq1
    except ImportError:
        raise RuntimeError('Install biopython: pip install biopython')
    records, errors, warnings = {}, [], []
    id_splits, seq_groups = defaultdict(set), defaultdict(list)
    for split in ('train', 'eval', 'test'):
        listing = args.root / 'data/splits' / f'{split}_af{args.suffix}.txt'
        if not listing.exists():
            raise RuntimeError(f'Missing {listing}. Use --suffix "" for names without suffix.')
        ids = [s.strip() for s in listing.read_text().splitlines() if s.strip()]
        repeats = sum(n - 1 for n in Counter(ids).values())
        if repeats:
            errors.append(f'{split}: {repeats} duplicate IDs within list')
        loaded = 0
        for pid in dict.fromkeys(ids):
            id_splits[pid].add(split)
            try:
                m = re.fullmatch(r'AF_AF(.+?)(F\d+)_\d+', pid)
                if not m:
                    raise ValueError('invalid ID')
                paths = list((args.root / 'structures' / split / 'af').glob(f'AF-{m[1]}-{m[2]}-model_v*.cif'))
                if len(paths) != 1:
                    raise ValueError(f'expected 1 CIF version, found {len(paths)}')
                structure = MMCIFParser(QUIET=True).get_structure('s', str(paths[0]))
                chains = [[r for r in chain if r.id[0] == ' ' and 'CA' in r]
                          for chain in next(structure.get_models())]
                chains = [c for c in chains if c]
                if len(chains) != 1:
                    raise ValueError('expected one chain containing CA atoms')
                seq = ''.join(seq1(r.resname, custom_map={'MSE':'M'}, undef_code='X') for r in chains[0])
                key = f'p{len(records)}'
                records[key] = (split, pid, seq)
                seq_groups[seq].append((split, pid))
                loaded += 1
                if 'X' in seq:
                    warnings.append(f'{split}/{pid}: {seq.count("X")} unknown residues')
            except Exception as exc:
                errors.append(f'{split}/{pid}: {exc}')
        print(f'{split}: loaded {loaded}/{len(ids)}', flush=True)
    overlaps = sum(len(s) > 1 for s in id_splits.values())
    duplicate_pairs = []
    for group in seq_groups.values():
        duplicate_pairs.extend((a,b) for a,b in itertools.combinations(group, 2) if a[0] != b[0])
    print(f'IDs shared across splits: {overlaps}')
    print(f'Identical sequence pairs across splits: {len(duplicate_pairs)}')
    for a,b in duplicate_pairs[:5]:
        print(f'  IDENTICAL: {a[0]}/{a[1]} <-> {b[0]}/{b[1]}')
    for msg in (errors + warnings)[:10]:
        print('  NOTE:', msg)
    print(f'Load/list errors: {len(errors)}; unknown-residue warnings: {len(warnings)}')
    missing = [x for x in ('blastp','makeblastdb') if not shutil.which(x)]
    if missing:
        raise RuntimeError('Similarity audit INCOMPLETE. Missing ' + ', '.join(missing) + '. Install: conda install -c bioconda blast')
    serious = overlaps > 0 or bool(duplicate_pairs) or bool(errors)
    broad_total, fragment_total, intermediate_total = 0, 0, 0
    print(f'Criteria: identity > {args.identity}%, coverage >= {args.coverage}% in BOTH proteins.')
    with tempfile.TemporaryDirectory(prefix='psp_audit_') as temporary:
        tmp = Path(temporary)
        fasta, db = {}, {}
        for split in ('train','eval','test'):
            fasta[split] = tmp / f'{split}.fa'
            fasta[split].write_text(''.join(f'>{k}\n{v[2]}\n' for k,v in records.items() if v[0] == split))
            if fasta[split].stat().st_size == 0:
                raise RuntimeError(f'No usable sequences for {split}; audit INCOMPLETE')
        for split in ('train','eval'):
            db[split] = tmp / f'db_{split}'
            command(['makeblastdb','-in',fasta[split],'-dbtype','prot','-out',db[split]])
        for qsplit, tsplit in [('eval','train'),('test','train'),('test','eval')]:
            print(f'Comparing {qsplit} -> {tsplit}...', flush=True)
            n = sum(v[0] == tsplit for v in records.values())
            text = command(['blastp','-query',fasta[qsplit],'-db',db[tsplit],'-task','blastp',
                '-outfmt','6 qseqid sseqid pident qlen slen qstart qend sstart send length evalue',
                '-evalue','1e-5','-max_target_seqs',str(n),'-max_hsps','1',
                '-num_threads',str(args.threads)])
            broad, fragments, intermediate, examples = set(), set(), set(), []
            for line in text.splitlines():
                q,s,ident,ql,sl,qs,qe,ss,se,length,e = line.split()
                ident = float(ident)
                ca = 100*(abs(int(qe)-int(qs))+1)/int(ql)
                cb = 100*(abs(int(se)-int(ss))+1)/int(sl)
                if ident > args.identity and min(ca,cb) >= args.coverage:
                    broad.add(q)
                    if len(examples) < 3:
                        examples.append(f'{records[q][1]} vs {records[s][1]}: identity {ident:.1f}%, coverage {ca:.1f}%/{cb:.1f}%')
                elif ident > args.identity and int(length) >= 50:
                    fragments.add(q)
                elif ident >= 30 and min(ca,cb) >= args.coverage:
                    intermediate.add(q)
            count = sum(v[0] == qsplit for v in records.values())
            print(f'  Above threshold, broad coverage: {len(broad)}/{count} ({100*len(broad)/count:.1f}%)')
            print(f'  Above threshold, partial match >=50 alignment columns: {len(fragments)} queries')
            print(f'  Broad matches at 30%-{args.identity:g}% identity: {len(intermediate)} queries (context only)')
            for ex in examples:
                print('   ', ex)
            broad_total += len(broad)
            fragment_total += len(fragments)
            intermediate_total += len(intermediate)
    print('\nVERDICT:')
    if serious or broad_total:
        print('CHECK REQUIRED: duplicates, input errors, or broad matches above the selected threshold were found.')
        print('This does not automatically invalidate the dataset; inspect the counts and examples above.')
    elif fragment_total or warnings:
        print('CAUTION: no broad above-threshold matches found, but partial matches or uncertain residues need a look.')
    else:
        print('No obvious problem found under these criteria. Reasonable first check for a student project.')
    print('This is a heuristic BLAST screen, not a guarantee of no leakage.')
    print('Coverage uses one local alignment span per pair; partial-domain screen uses an arbitrary 50-column minimum.')
    print('Sequences come from CA-bearing CIF residues. No persistent report files were created.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR / AUDIT INCOMPLETE: {exc}', file=sys.stderr)
        sys.exit(1)
