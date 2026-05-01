#!/usr/bin/env python3
"""Wrapper to gather metadata for sample points.

Run:
  python3 0b_gather_metadata.py

Input:
  data/sample_data_filtered.csv (must contain Latitude/Longitude)

Output:
  data/sample_data_filtered_full.csv

Pipeline (best-effort, Option A: do not fail whole run if a dataset is missing):
  1) CHELSA BioClim (bio1..bio19)
  2) Topography (DEM elev/slope/aspect + focal 100m/1000m)
  3) NDVI (Landsat8 2013-2018 greenestpixel max NDVI, 30 arc-sec)
  4) Land cover (ESA WorldCover 10m)
  5) Soil properties (SoilGrids pH 0-5cm mean)

Notes:
- /data is gitignored; you must provide rasters locally where required.
- Each part also writes a per-dataset CSV under data/<dataset>/.

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR, log_api_data

from src import part1_climate, part2_topography, part3_ndvi, part4_land_cover, part5_soil


INPUT_CSV = DATA_DIR / "sample_data_filtered.csv"
OUTPUT_CSV = DATA_DIR / "sample_data_filtered_full.csv"


def _safe_run(name: str, func, base_index) -> pd.DataFrame:
    try:
        df = func(INPUT_CSV)
        # normalize index
        df = df.reset_index(drop=True)
        if len(df) != len(base_index):
            raise RuntimeError(f"{name} returned {len(df)} rows, expected {len(base_index)}")
        return df
    except Exception as e:
        log_api_data([f"WARNING: {name} failed; filling columns with NA", f"Error: {repr(e)}"], script_name=Path(__file__).name)
        return pd.DataFrame(index=range(len(base_index)))


def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    base = pd.read_csv(INPUT_CSV).reset_index(drop=True)

    log_api_data(
        [
            f"Wrapper input: {INPUT_CSV}",
            f"Wrapper output: {OUTPUT_CSV}",
            "Mode: Option A (best-effort; dataset failures do not abort the run)",
        ],
        script_name=Path(__file__).name,
    )

    climate = _safe_run("CHELSA", part1_climate.run, base.index)
    topo = _safe_run("Topography", part2_topography.run, base.index)
    ndvi = _safe_run("NDVI", part3_ndvi.run, base.index)
    lc = _safe_run("Land cover", part4_land_cover.run, base.index)
    soil = _safe_run("Soil", part5_soil.run, base.index)

    full = pd.concat([base, climate, topo, ndvi, lc, soil], axis=1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
