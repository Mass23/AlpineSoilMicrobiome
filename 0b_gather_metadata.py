#!/usr/bin/env python3
"""Wrapper to gather metadata for sample points.

Run:
  python3 0b_gather_metadata.py

Input:
  data/sample_data_filtered.csv (must contain Latitude/Longitude)

Output:
  data/sample_data_filtered_full.csv

Pipeline:
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

from src import 0b_part1_climate as part1  # type: ignore
from src import 0b_part2_topography as part2  # type: ignore
from src import 0b_part3_ndvi as part3  # type: ignore
from src import 0b_part4_land_cover as part4  # type: ignore
from src import 0b_part5_soil as part5  # type: ignore


INPUT_CSV = DATA_DIR / "sample_data_filtered.csv"
OUTPUT_CSV = DATA_DIR / "sample_data_filtered_full.csv"


def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    base = pd.read_csv(INPUT_CSV)

    log_api_data([
        f"Wrapper input: {INPUT_CSV}",
        f"Wrapper output: {OUTPUT_CSV}",
    ], script_name=Path(__file__).name)

    climate = part1.run(INPUT_CSV)
    topo = part2.run(INPUT_CSV)
    ndvi = part3.run(INPUT_CSV)
    lc = part4.run(INPUT_CSV)
    soil = part5.run(INPUT_CSV)

    full = pd.concat([base, climate, topo, ndvi, lc, soil], axis=1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
