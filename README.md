# Protein Structure Prediction (PSP) — Team 2 (AFDB)

Repozytorium projektu z przedmiotu Bioinformatyka / Uczenie Maszynowe w Biologii.
Projekt skupia się na predykcji trójwymiarowych struktur białkowych na bazie modeli z bazy **AlphaFold DB (AFDB)**.

---

## Struktura Repozytorium

```
projekt/
├── data/
│   ├── splits/                     # Zdefiniowane zbiory (listy ID białek)
│   │   ├── train_af.txt            # 4000 białek (train, klastry 50% lub 30%)
│   │   ├── eval_af.txt             # 500 białek (walidacja)
│   │   ├── test_af.txt             # 500 białek (test)
│   │   └── *_30.txt                # Wersje ze ścisłym klastrowaniem 30%
│   └── raw/                        # Katalog na surowe pliki klastrów RCSB (gitignored)
│
├── notebooks/
│   └── 01_dataset_preparation_afdb.ipynb  # Kompleksowy notebook przygotowania danych
│
├── src/                            # Współdzielone moduły Pythona
│   ├── __init__.py
│   ├── data.py                     # Ładowanie splitów, tłumaczenie ID, szybki parser CIF
│   ├── metrics.py                  # Metryki geometryczne (RMSD, dRMSD, TM-score)
│   └── models.py                   # Szkielety modeli i architektur (Meeting 3 & 6)
│
├── scripts/                        # Samodzielne skrypty CLI
│   ├── split_clusters.py           # Generator podziału klastrów (flagi --identity 30/50)
│   ├── download_structures.sh      # Pobieranie struktur CIF z AlphaFold DB (obsługa 429)
│   └── extract_chain.py            # Skrypt pomocniczy (Team 1 PDB)
│
├── structures/                     # Pobrane pliki .cif (~1.6 GB, gitignored, współdzielone dyskiem)
│   ├── train/af/
│   ├── eval/af/
│   └── test/af/
│
├── requirements.txt                # Zależności bibliotek w Pythonie
├── .gitignore                      # Ignorowanie dużych struktur, wag modeli i cache
└── README.md                       # Dokumentacja projektu
```

---


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

### 2. Dostęp do struktur białkowych (~1.6 GB)
Pliki `.cif` ze względu na rozmiar nie są trzymane w GitHube.
* **Opcja A (Zalecana):** Pobierz spakowane archiwum `structures_af.tar.gz` ze wspólnego Google Drive'a i rozpakuj w głównym folderze:
  ```bash
  tar -xzvf structures_af.tar.gz
  ```
* **Opcja B (Pobranie skryptem):**
  ```bash
  ./download_structures.sh eval af
  ./download_structures.sh test af
  ./download_structures.sh train af
  ```

### 3. Uruchomienie notebooka
```bash
jupyter notebook notebooks/01_dataset_preparation_afdb.ipynb
```

