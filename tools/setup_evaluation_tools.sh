#!/usr/bin/env bash
# Setup script for Meeting 4 evaluation tools:
# 1. Zhang Group TM-score (C++ compiled natively)
# 2. OpenStructure (ost) via micromamba/conda

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
mkdir -p "$TOOLS_DIR"

echo "=== 1. Downloading & compiling Zhang Group TM-score ==="
if [ ! -f "$TOOLS_DIR/TMscore.cpp" ]; then
    curl -fsSL https://zhanggroup.org/TM-score/TMscore.cpp -o "$TOOLS_DIR/TMscore.cpp"
fi
g++ -O3 -ffast-math -o "$TOOLS_DIR/TMscore" "$TOOLS_DIR/TMscore.cpp"
chmod +x "$TOOLS_DIR/TMscore"
echo "✓ TMscore ready at: $TOOLS_DIR/TMscore"

echo ""
echo "=== 2. Installing OpenStructure (ost) into .ost_env ==="
if [ ! -x "$ROOT_DIR/.ost_env/bin/ost" ]; then
    if command -v micromamba >/dev/null 2>&1; then
        micromamba create -y -p "$ROOT_DIR/.ost_env" -c conda-forge -c bioconda python=3.11 openstructure
    elif command -v conda >/dev/null 2>&1; then
        conda create -y -p "$ROOT_DIR/.ost_env" -c conda-forge -c bioconda python=3.11 openstructure
    else
        echo "WARNING: Neither micromamba nor conda found."
        echo "Install micromamba (e.g. 'brew install micromamba') or use OpenStructure Docker image:"
        echo "  docker pull registry.scicore.unibas.ch/schwede/openstructure:latest"
        exit 1
    fi
fi
echo "✓ OpenStructure ready at: $ROOT_DIR/.ost_env/bin/ost"
