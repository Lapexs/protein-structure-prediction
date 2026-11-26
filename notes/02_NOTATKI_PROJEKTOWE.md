# Notatki projektowe: część AlphaFold DB (AF)

Plik zbiera informacje organizacyjne i plan prac, które towarzyszą notebookowi `02_end_to_end_skeleton.ipynb`. Sam notebook zawiera rozwiązanie i jego opis.

## Podział pracy w zespole

- Dane są podzielone na dwie ścieżki: jedna osoba odpowiada za zbiór eksperymentalny **PDB**, a notebook realizuje potok dla zbioru **AlphaFold DB (AF)**.
- Pobrane pliki to `.cif` (`AF-...-model_v6.cif`) w `structures/{train,eval,test}/af/`.
- Wspólnym formatem wyjściowym obu ścieżek jest wwPDB Format v3.3 (rekordy `ATOM` dla atomów Cα), co ułatwi późniejsze porównanie wyników.

## Zakres Meeting 3 (szkielet end-to-end)

Harmonogram: *end-to-end skeleton with working input/output*. Notebook realizuje to w krokach:

1. Wczytanie multi-protein FASTA przez `skbio.io.read()` (Krok 2).
2. Kodowanie do `(B, D, N)` (B = 16, D = 1024, N = 22) i maska `(B, D)` (Krok 3).
3. Model PyTorch zwracający `(L, 3)` (Krok 4: atrapa `DummyStructurePredictor`).
4. Zapis PDB: 1 rekord `ATOM` na resztę, 0 rekordów dla paddingu (Kroki 5 i 6).
5. Prawdziwe dane z `.cif` i dowód trenowalności na dodatkowym modelu `TrainableStructurePredictor` ze stratą dRMSD (Kroki 7–10).

Wyniki z uruchomienia na 4 białkach testowych (Kroki 1–6):

| Białko | Długość | Zapisane rekordy CA |
|---|---|---|
| AF-A0A075B695-F1 | 113 | 113 |
| AF-A0A077ZFK1-F1 | 359 | 359 |
| AF-A0A1D6IU37-F1 | 1112 | 1024 (ucięte do D) |
| AF_TEST_UNK_F1 | 25 (z symbolem `X`) | 25 (`UNK`) |

## Zmiany względem pierwotnej wersji notebooka

- Atrapa `DummyStructurePredictor` dostała parametry skali całości (`coord_scale`, `proj_weight`) i lokalnej geometrii (`mode`, `bond_length`, `bond_angle_deg`, `dihedral_range_deg`). Wartości domyślne odtwarzają pierwotne zachowanie, czyli niezależne losowe punkty ze skalą 15 Å.
- Dodano Kroki 7–10: czytanie `.cif` (sekwencja i współrzędne Cα) z list `data/splits`, model `TrainableStructurePredictor`, stratę dRMSD, dowód trenowalności i zapis PDB z modelu uczonego.
- Dodano funkcję `cifs_to_fasta`, która zapisuje sekwencje z `.cif` do FASTA.
- Podział pracy, odwołania do spotkań i roadmapa zostały przeniesione z notebooka do tego pliku.

## Dane i konfiguracja

- Zmienna `SPLIT_SUFFIX` (Krok 7) wybiera wariant list: `"_30"` to wariant standardowy (4000 / 500 / 500), `""` to wariant duży (40 000 / 5 000 / 5 000).
- Białka są wybierane z list w `data/splits` za pomocą `load_split` i `af_id_to_filename` z `src/data.py`.
- Parametry dowodu trenowalności (Krok 9): `PROOF_N`, `PROOF_MAX_LEN`, `PROOF_STEPS`, `PROOF_LR`, `PROOF_SEED`.
- Pliki wynikowe: `predictions/meeting3_pdb` (atrapa), `predictions/trainable_pdb` (model uczony), `predictions/trainability_loss.png` (wykres straty).

## Wiarygodne wartości parametrów atrapy

| Parametr | Wartość w notebooku | Wartość wiarygodna fizycznie |
|---|---|---|
| `coord_scale` (tryb `noise`) | 15.0 Å | std ≈ Rg / √3, gdzie Rg ≈ 2.2·L^0.38 Å (przybliżenie empiryczne). Dla L = 100: ok. 7 Å, L = 300: ok. 11 Å, L = 1000: ok. 17 Å |
| `bond_length` (tryb `chain`) | 3.8 Å | 3.8 Å (ok. 2.9 Å dla cis-proliny) |
| `bond_angle_deg` | 110° | ok. 85–130°; helisa alfa ok. 91°, nić beta ok. 120–125° |
| `dihedral_range_deg` | 180° | 180° dla swobodnego łańcucha; helisa ok. +50°, nić beta ok. −170°/+190° |

Tryb `noise` losuje każdy atom niezależnie, a tryb `chain` buduje łańcuch ze stałą długością wiązania i kątem między kolejnymi Cα.

## Dalszy plan prac

1. **Meeting 4: framework ewaluacyjny.** Porównanie przewidzianych Cα `(L, 3)` z referencją z `.cif` AlphaFold DB. Metryki: RMSD (Kabsch), TM-score, GDT_TS, lDDT.
2. **Meeting 5: metody bazowe.** ESM3, ESMFold2 i podobne na zbiorze walidacyjnym i testowym (`eval_af`, `test_af`).
3. **Meeting 6: implementacja modelu.** Architektura docelowa trenowana na `train_af`, w miejsce modelu szkieletowego.
4. **Meeting 7: prezentacja końcowa (20 min).** Porównanie wyników ze ścieżki AlphaFold DB i ścieżki PDB.
