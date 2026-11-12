# Protein Structure Prediction (PSP) — Team 2 (AFDB)

Repozytorium projektu z przedmiotu Bioinformatyka / Uczenie Maszynowe w Biologii.
Projekt skupia się na predykcji trójwymiarowych struktur białkowych na bazie modeli z bazy **AlphaFold DB (AFDB)**.

---

## 📌 Harmonogram Projektu (Milestones)

* [x] **Meeting 1:** Overview of the projects; team assignment (Team 2: AFDB).
* [x] **Meeting 2:** Train / test dataset preparation (klastrowanie sekwencji 30% i 50%, podział 4000/500/500, pobieranie struktur CIF).
* [ ] **Meeting 3:** End-to-end skeleton with working input/output (pipeline: sekwencja $\to$ dummy model $\to$ współrzędne 3D $\to$ CIF).
* [ ] **Meeting 4:** Adding evaluation framework (metryki: RMSD, TM-score, GDT_TS, lDDT).
* [ ] **Meeting 5:** Adding baseline methods (benchmark z ESMFold / ESM3).
* [ ] **Meeting 6:** Model implementation (docelowa architektura sieci grafowej / transformera i trening).
* [ ] **Meeting 7:** Final presentation (20 min. podsumowania wyników, analizy błędów i wizualizacji 3D).

---

## 📂 Struktura Repozytorium

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

---

## 👥 Podział Pracy w Zespole (3 osoby)

1. **Osoba 1: Data Engineering & Preprocessing**
   * Prowadzi moduł `src/data.py` (ekstrakcja cech aminokwasowych, współrzędnych $C_\alpha$, macierzy odległości).
   * Buduje klasy `torch.utils.data.Dataset` oraz `DataLoader`.
2. **Osoba 2: Architektura Modelu & Trening**
   * Prowadzi moduł `src/models.py`.
   * Odpowiada za szkielet wejście/wyjście (Meeting 3), integrację baseline'u ESMFold (Meeting 5) oraz docelowy model (Meeting 6).
3. **Osoba 3: Ewaluacja, Benchmark & Prezentacja**
   * Prowadzi moduł `src/metrics.py` (RMSD, TM-score, lDDT).
   * Odpowiada za logowanie do Weights & Biases, analizę biologiczną błędów (*error analysis*) oraz przygotowanie prezentacji końcowej (Meeting 7).
