# Reproducibility archive — biophysics of Tumour Treating Fields

Open code and generated figures associated with a review manuscript currently
under journal peer review. The manuscript itself is **not** distributed via this
repository; it will be linked from the journal DOI upon publication.

All figures are computed from established analytical biophysical models
(single-shell Maxwell–Wagner dielectric cell model; Schwan induced
transmembrane-potential equation; concentric-shell core-field solution;
Clausius–Mossotti / dielectrophoresis theory) using only openly published
dielectric parameters cited in the manuscript. No experimental or patient data
are used.

## Layout
- `code/make_figures.py` — single script that reproduces the quantitative Figures 2–5 (it also writes simple placeholders for Figures 1 and 6, which appear in the article as schematic illustrations)
- `figures/` — generated figures (PDF)

## Reproduce
```bash
pip install numpy scipy matplotlib
python code/make_figures.py   # writes figures/*.pdf (run from repo root)
```

## License
Released under the MIT License (see `LICENSE`).
