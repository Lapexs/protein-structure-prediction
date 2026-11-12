#!/usr/bin/env bash
# Usage: ./download_structures.sh train pdb
#        ./download_structures.sh eval  af

SPLIT=$1   # train / eval / test
DB=$2      # pdb / af
OUT="structures/${SPLIT}/${DB}"
mkdir -p "$OUT"

SPLIT_FILE="${SPLIT}_${DB}.txt"
if [[ ! -f "$SPLIT_FILE" && -f "data/splits/${SPLIT}_${DB}.txt" ]]; then
    SPLIT_FILE="data/splits/${SPLIT}_${DB}.txt"
fi

if [[ ! -f "$SPLIT_FILE" ]]; then
    echo "Error: Split file '$SPLIT_FILE' not found (checked '.' and 'data/splits/')."
    exit 1
fi

if [[ "$DB" == "pdb" ]]; then
    awk -F'_' '{print "https://files.rcsb.org/download/"$1".cif"}' "$SPLIT_FILE" \
        | sort -u | wget -nc -i - -P "$OUT"

elif [[ "$DB" == "af" ]]; then
    python3 -c "
import re
for line in open('$SPLIT_FILE'):
    m = re.match(r'AF_AF(.+?)(F\d+)_\d+$', line.strip())
    if m:
        print(f'https://alphafold.ebi.ac.uk/files/AF-{m.group(1)}-{m.group(2)}-model_v6.cif')
" | sort -u | wget -nc --timeout=15 --wait=0.35 --retry-on-http-error=429 --waitretry=15 --tries=10 -i - -P "$OUT"
fi
