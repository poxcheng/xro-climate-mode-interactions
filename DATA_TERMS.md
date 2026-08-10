# Data terms, attribution, and citations

The `data/` directory contains **author-derived** files (subsets, indices, regridded or averaged
fields) computed from the upstream products below. Each upstream product retains its own terms;
the authors' original additions (derived indices, XRO operator fits, caches) are provided under
CC BY 4.0 to the extent the authors hold rights in them.

| file group | upstream product | terms / license | required attribution & citation |
|---|---|---|---|
| `ERA5_*.nc` | ERA5 monthly means, Copernicus Climate Change Service (C3S) Climate Data Store | [Copernicus Products licence](https://cds.climate.copernicus.eu/licences/licence-to-use-copernicus-products) — reproduction/distribution/adaptation permitted with attribution | "Contains modified Copernicus Climate Change Service information (2026). Neither the European Commission nor ECMWF is responsible for any use of this information." Cite Hersbach et al. (2020, QJRMS). |
| `XRO_indices_oras5_extended_2025.nc` | ORAS5 ocean reanalysis (ECMWF, via C3S CDS) + XRO indices of Zhao et al. (2024) | Copernicus Products licence (as above) | As above; additionally cite Zhao et al. (2024, Nature 630, 891–898). 1979–2024 verified identical (r = 1.000000) to the official ORAS5-based XRO indices; 2025 extension computed by the authors from ORAS5. |
| `pocheng_tropical_mean_SST_XRO/` | CMIP5 (Taylor et al. 2012) and CMIP6 (Eyring et al. 2016) historical simulations, ESGF | CMIP5 terms of use; CMIP6 CC BY 4.0 ([terms](https://pcmdi.llnl.gov/CMIP6/TermsOfUse/)) | "We acknowledge the World Climate Research Programme's Working Group on Coupled Modelling, which coordinated and promoted CMIP." Model list in Supplementary Table 1. Files are author-computed XRO operator fits, not raw model output. |
| `realtime_oisst_sst_indices.nc` | NOAA OISST v2.1 (NCEI) | Public (U.S. Government work); [product page](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) | Cite Huang et al. (2021, J. Climate). |
| `realtime_tao_pmel_wwv.nc` | PMEL/TAO equatorial Pacific warm water volume (NOAA PMEL) | Public; [product page](https://www.pmel.noaa.gov/tao/wwv/data/) | Acknowledge the TAO Project Office of NOAA/PMEL. |
| `reference_cache/*` | Author-generated engine outputs (from the inputs above) | CC BY 4.0 | Cite this paper. |
| `beta_from_patterns.nc` | Author-derived (legacy β-projection file; not read in the standard `USE_STANDARD_BETA=True` configuration; retained for completeness) | CC BY 4.0 | Cite this paper. |
