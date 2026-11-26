# Protein Structure Prediction (PSP) — AlphaFold DB (AF)

Repozytorium projektu z przedmiotu Bioinformatyka / Uczenie Maszynowe w Biologii.
Projekt skupia się na predykcji trójwymiarowych struktur białkowych na bazie modeli z bazy **AlphaFold DB (AFDB)**.

---

## 📂 Struktura Repozytorium

```
projekt/
├── data/
│   ├── splits/                     # Zdefiniowane zbiory (listy ID białek)
│   │   ├── train_af.txt            # 40 000 białek (Wariant Duży, klastry 30%)
│   │   ├── eval_af.txt             # 5 000 białek (walidacja)
│   │   ├── test_af.txt             # 5 000 białek (test)
│   │   └── *_30.txt                # Wersja standardowa (4000 / 500 / 500)
│   ├── samples/                    # Przykładowe pliki FASTA do testowania predyktora
│   │   └── sample_multi.fasta      # Testowy multi-protein FASTA (różne długości, X, >1024 AA)
│   ├── processed/                  # Binarne cache tensorowe PyTorch (*.pt, gitignored)
│   └── raw/                        # Katalog na surowe pliki klastrów RCSB (gitignored)
│
├── notebooks/
│   ├── 01_dataset_preparation_afdb.ipynb  # Przygotowanie i pobieranie danych (Meeting 2)
│   └── 02_end_to_end_skeleton.ipynb       # Szkielet end-to-end wejście/wyjście (Meeting 3)
│
├── src/                            # Współdzielone moduły Pythona
│   ├── __init__.py
│   ├── predictor.py                # Szablon predyktora (scikit-bio, tensory (B, D, N), maska, format wwPDB)
│   ├── data.py                     # Ładowanie splitów, dataset PyTorch (cache/CIF), parser, eksport PDB
│   ├── metrics.py                  # Metryki geometryczne (dRMSD, Kabsch RMSD, contact precision)
│   └── models.py                   # Model ProteinStructureSkeleton i funkcja straty dRMSDLoss
│
├── scripts/                        # Samodzielne skrypty CLI
│   ├── predict_structure.py        # CLI predyktora FASTA -> wwPDB (Meeting 3)
│   ├── run_skeleton.py             # Uruchomienie demonstracji szkieletu end-to-end (dRMSD loss)
│   ├── split_clusters.py           # Generator podziału klastrów (--train, --eval, --test, --identity)
│   ├── download_structures_fast.py # Równoległe wielowątkowe pobieranie CIF (obsługa HTTP 429 i resume)
│   ├── build_dataset_cache.py      # Budowanie kompaktowego cache tensorowego .pt z plików CIF
│   ├── download_structures.sh      # Pobieranie struktur skryptem bash
│   └── extract_chain.py            # Skrypt pomocniczy (dla struktur eksperymentalnych PDB)
│
├── tests/                          # Testy jednostkowe
│   └── test_meeting3_skeleton.py   # Testy parsowania FASTA, kształtów tensorów (B,D,N), maskowania i wwPDB
│
├── predictions/                    # Przykładowe wyjścia modelu (pliki .pdb i wykresy)
│   ├── AF_AFB1JDR0F1_1_predicted.pdb
│   └── distance_matrix_comparison.png
│
├── structures/                     # Pobrane pliki .cif (~1.6 GB, gitignored, współdzielone dyskiem)
│   ├── train/af/
│   ├── eval/af/
│   └── test/af/
│
├── requirements.txt                # Zależności bibliotek w Pythonie
├── pyproject.toml                  # Konfiguracja środowiska
├── .gitignore                      # Ignorowanie dużych struktur, wag modeli i cache
└── README.md                       # Dokumentacja projektu
```

---

## 🚀 Szybki Start (Quickstart dla członków grupy)

### 1. Klonowanie repozytorium i środowisko
```bash
git clone https://github.com/Lapexs/protein-structure-prediction.git
cd protein-structure-prediction

# Utworzenie i aktywacja środowiska wirtualnego:
python3 -m venv .venv
source .venv/bin/activate

# Instalacja zależności:
pip install -r requirements.txt
```

### 2. Dostęp do struktur białkowych i Cache
Pliki `.cif` ze względu na duży rozmiar nie są wrzucane do Gita.

* **Pobieranie struktur (wielowątkowy downloader z wznawianiem i obsługą rate-limit 429):**
  ```bash
  python scripts/download_structures_fast.py --split eval --workers 8
  python scripts/download_structures_fast.py --split test --workers 8
  python scripts/download_structures_fast.py --split train --workers 8
  ```

* **Budowanie binarnego cache PyTorch (`.pt`):**
  Zamienia pobrane pliki CIF na kompaktowe tensory w `data/processed/` (~150-200 MB dla 50k białek zamiast ~15 GB CIF), przyspieszając ładowanie w `Dataset` do ułamka sekundy:
  ```bash
  python scripts/build_dataset_cache.py --split all
  ```

### 3. Uruchomienie predyktora FASTA -> PDB (Meeting 3 Skeleton Template)
* **Wczytanie FASTA za pomocą `scikit-bio` i generowanie struktur PDB:**
  ```bash
  python scripts/predict_structure.py --fasta data/samples/sample_multi.fasta --out-dir predictions/meeting3_pdb
  ```
  Przetwarza multi-protein FASTA na tensory `(B, D, N)` z maską `(B, D)` i generuje pliki `.pdb` z atomami $C_\alpha$ ściśle według specyfikacji wwPDB Format v3.3 (brak wpisów dla paddingu).

* **Uruchomienie testu szkieletu treningowego (dRMSD loss & sanity check):**
  ```bash
  python scripts/run_skeleton.py
  ```

* **W Jupyter Notebooku:**
  ```bash
  jupyter notebook notebooks/02_end_to_end_skeleton.ipynb
  ```

* **Uruchomienie testów jednostkowych:**
  ```bash
  python tests/test_meeting3_skeleton.py
  ```
