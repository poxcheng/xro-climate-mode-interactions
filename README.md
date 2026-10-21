# Reproduction package — "Climate mode interactions shape El Niño-driven tropical ocean and atmospheric warming"

Po-Cheng Chen, Fei-Fei Jin, Sen Zhao (University of Hawaiʻi at Mānoa)

This package reproduces **all ten main-text figures** of the manuscript from upstream inputs, and, via `revision_figs/`, the figures added or redesigned in the first revision (Fig. 1, Fig. 10, and Supplementary Figs. 1-4).

## Contents

```
Full_code_reproduce_v1.ipynb      # the complete reproduction notebook (run top-to-bottom)
environment.yml                   # pinned conda environment (tested)
requirements.txt                  # pinned pip environment (tested)
data/                             # all required inputs (~825 MB) — see data_manifest.tsv
  ├── reference_cache/            #   author-generated engine caches (never read by the notebook)
  └── pocheng_tropical_mean_SST_XRO/  # per-model CMIP5/6 XRO operator fits (88 models, Supplementary Data 1)
figures/                          # output directory (created/filled by the notebook)
data_manifest.tsv                 # SHA-256 checksums + provenance for every data file
DATA_TERMS.md                     # per-source data licenses, attribution, and citations
LICENSE                           # MIT (applies to original code in this package only)
CITATION.cff
revision_figs/                    # standalone scripts for the figures added in revision R1 (see below)
```

## Quick start

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate xro-repro
jupyter lab Full_code_reproduce_v1.ipynb
```

Run all cells in order. Paths are relative: the notebook expects `./data` next to it and writes
figures to `./figures` (override with the `XRO_DATA_DIR` / `XRO_FIG_DIR` environment variables).
On Google Colab, the first cells pip-install `cftime` and `XRO` (pinned to senclimate/XRO v1.0.2)
and mount Drive; locally, use the pinned environment instead.

## What gets recomputed

Both engines run in reproduction mode (`FORCE_RECOMPUTE = True`):

| stage | runtime (verified clean run, Apple Silicon, Python 3.13) |
|---|---|
| Observed replay engine (Sec. 2) | ≈ 1 min |
| 88-model CMIP hybrid engine (Sec. 9) | ≈ 9 min |
| All remaining figure cells | ≈ 2 min |
| **Full notebook, top to bottom** | **≈ 12 min** |

The supplied `data/reference_cache/` files are **never read**. Recomputed results are written to
`./outputs_cache/*_repro*` files, which can be diffed against the reference caches.
The forecast ensemble (Fig. 10) uses a fixed random seed and is deterministic.

**Verification (clean-room run, 2026-08-10, macOS/Python 3.13, pinned environment):** all 30 cells
executed top-to-bottom with no errors; recomputed engine outputs agree with the reference caches to
max |diff| ≤ 5.3e-08 K (observed replay, all arrays) and ≤ 1.9e-06 K (CMIP engine, 6,090 numeric
blocks across all 2,024 model-experiment series); the regenerated figures are visually identical to
the manuscript figures.

## Expected outputs

Ten figures matching main-text Figs. 1–10 (Sections 4–8 and 10–13 of the notebook; Section 11
produces both Figs. 7 and 8). Figs. 1–5 render inline in the notebook; Figs. 6–10 are additionally
written to `./figures` as PNG (Fig. 10 also as PDF). Rendered outputs from a complete author run
are stored in the notebook itself for reference.

## Verify data integrity

```bash
cd data && shasum -a 256 -c ../data_manifest.tsv
```

(The manifest's first two columns are `sha256  path`; the remaining provenance columns are ignored by `shasum -c`.)

## Data provenance & terms

See `DATA_TERMS.md`. Upstream products: ERA5 (Copernicus CDS), ORAS5 (ECMWF/CDS), CMIP5/CMIP6
(ESGF), NOAA OISST v2.1 (NCEI), PMEL/TAO warm water volume (NOAA PMEL). The XRO model code is
by Zhao et al. (2024), https://github.com/senclimate/XRO (v1.0.2), archived at
https://doi.org/10.5281/zenodo.10681114.

## Contact

Po-Cheng Chen — pocheng@hawaii.edu

## Revision figures (R1)

The first-revision figures are produced by standalone scripts in `revision_figs/`. Each script
resolves the input directory from the `XRO_DATA_DIR` environment variable (default `./data`
next to this README, the same data as the notebook) and writes its PNG to the current working
directory.

| script | manuscript item | inputs |
|---|---|---|
| `fig01_timeseries.py` | Fig. 1 | `data/` + `intermediates/ltt_depth_check.pkl` |
| `fig10_forecast.py` | Fig. 10 and Supplementary Fig. 3 | `data/` (runs `forecast_engine.py`; fixed seed, deterministic) |
| `si_fig1_obs_support.py` | Supplementary Fig. 1 | `data/` + `intermediates/` |
| `si_fig2_removal.py` | Supplementary Fig. 2 | `intermediates/rm1d.pkl` |
| `si_fig4_mode_contrib.py` | Supplementary Fig. 4 | `intermediates/mode_contrib.pkl` |

`forecast_engine.py` is the self-contained real-time forecast engine (OISST and WWV initial
conditions, operational XRO configuration, 1000-member red-noise ensemble with a fixed seed).
`revision_figs/intermediates/` ships the small pickled intermediates used by the scripts;
`revision_figs/make_intermediates/` contains the scripts that recompute every intermediate from
`data/` (for `rm1d.py`, by re-running the CMIP hybrid engine of the main notebook with the
coupling-removal cases), so the full chain from upstream inputs to every figure is reproducible.
