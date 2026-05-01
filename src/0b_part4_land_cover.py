#!/usr/bin/env python3
"""Land cover point sampling (ESA WorldCover 10m).

Workflow:
- User downloads ESA WorldCover 10m GeoTIFF (global mosaic or region subset)
- Place it locally under:
    data/land_cover/esa_worldcover_10m.tif
- Script samples class code per point.
- Adds class name column.

Output:
  data/land_cover/land_cover_points.csv

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_DIR, log_api_data, load_points, sample_raster_at_points, sha256_file


WORLDCOVER_TIF = DATA_DIR / "land_cover" / "esa_worldcover_10m.tif"

# ESA WorldCover v100/v200 class codes are similar; keep mapping broad.
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def run(input_csv: Path = DATA_DIR / "sample_data_filtered.csv") -> pd.DataFrame:
    lc_dir = DATA_DIR / "land_cover"
    lc_dir.mkdir(parents=True, exist_ok=True)

    if not WORLDCOVER_TIF.exists():
        raise FileNotFoundError(
            f"Missing ESA WorldCover raster: {WORLDCOVER_TIF}\n"
            "Download ESA WorldCover 10m and place it at that path."
        )

    df = load_points(input_csv)
    valid = df["__valid_latlon"].to_numpy()
    lats = df["Latitude"].to_numpy(float)
    lons = df["Longitude"].to_numpy(float)

    log_api_data(
        [
            "Land cover: ESA WorldCover 10m (user-provided raster)",
            f"Path: {WORLDCOVER_TIF}",
            f"sha256: {sha256_file(WORLDCOVER_TIF)}",
        ],
        script_name=Path(__file__).name,
    )

    code = np.full(len(df), np.nan)
    code[valid] = sample_raster_at_points(WORLDCOVER_TIF, lons[valid], lats[valid])
    code_int = pd.Series(code).round().astype("Int64")
    name = code_int.map(WORLDCOVER_CLASSES).astype("string")

    out = pd.DataFrame(
        {
            "esa_worldcover_code": code_int,
            "esa_worldcover_class": name,
        },
        index=df.index,
    )

    out_path = lc_dir / "land_cover_points.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run()
