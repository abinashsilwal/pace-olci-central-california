# PACE OCI and Sentinel-3 OLCI along the central California coast

Code for the comparison of PACE OCI and Sentinel-3 OLCI remote-sensing reflectance and chlorophyll-a, both processed with NASA's `l2gen`, and for their evaluation against in situ observations from the NASA SeaBASS archive.

## Contents

| Path | Purpose |
|---|---|
| `config.yaml` | Study area, selected days, bands, quality-control limits, grid and bootstrap settings |
| `run.py` | Data discovery, download, gridding and basic statistics |
| `run_final.py` | Final analysis: pooled and stratified statistics, in situ evaluation, sensitivity tests, tables, figures and the numbers used in the text |
| `fetch_srf.py` | Downloads the OLCI spectral response functions |
| `pipeline/` | Processing and analysis modules |
| `scripts/` | `run_insitu_days.py` grids the days with in situ stations (dates in `insitu_days.txt`) |
| `results/` | Derived result tables behind the paper's tables and figures |
| `fonts/` | Typeface used in the figures (Source Sans 3, SIL Open Font License) |

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. Downloading the satellite granules needs a free NASA Earthdata account (`earthaccess` reads credentials from the environment variables `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or from `~/.netrc`).

## Usage

Run all commands from the repository root.

```
pip install -r requirements.txt
python fetch_srf.py                 # OLCI spectral response functions -> data/srf
python run.py download              # Level-2 granules for the days listed in config.yaml -> data/raw
python run.py process               # quality control, spectral matching and gridding -> data/processed
python scripts/run_insitu_days.py   # same processing for the days with in situ stations -> data/processed_insitu
python run_final.py all             # statistics, tables, figures and numbers -> outputs/
```

`run_final.py` also accepts `core`, `insitu`, `sens` or `assemble` to run one part. The in situ stations are read from the SeaBASS files listed in the paper; `pipeline/insitu.py` describes the expected input files under `data/insitu`.

## Data

- PACE OCI Level-2 products (version 3.2) and Sentinel-3A/3B OLCI Level-2 full-resolution products (version R2022.0): NASA Ocean Biology Distributed Active Archive Center, https://oceancolor.gsfc.nasa.gov
- In situ data: NASA SeaBASS, https://seabass.gsfc.nasa.gov (dataset DOIs are given in the paper)
- Coastline: Natural Earth 1:10m land polygons

## Citation

Silwal, A. et al. (2026). PACE OCI and Sentinel-3 OLCI show consistent chlorophyll-a but band-dependent reflectance offsets along the central California coast. Manuscript in preparation.

## License

MIT. See `LICENSE`. The figure typeface in `fonts/` is distributed under the SIL Open Font License.
