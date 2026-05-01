#!/usr/bin/env python3
"""CHELSA BioClim (bio1..bio19) point sampling.

Behavior:
- Expects user to have CHELSA BioClim GeoTIFFs available locally under data/chelsa/
  OR uses remote streaming URLs (CHELSA V2.1) if available.
- Samples all 19 variables at point locations.
- Writes data/chelsa/chelsa_bioclim_points.csv (local, gitignored).
- Logs URLs/versions and (if local) file SHA256 to data/api_data_log.txt.

We mirror the URL pattern used in Mass23/CrystalBall `code/a_CreateClimate.py`.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_DIR, log_api_data, load_points, sample_raster_at_points, sha256_file


BIOVARS = [f"bio{i}" for i in range(1, 20)]

# CHELSA V2.1 baseline bioclim path pattern from the CrystalBall example
CHELSA_BASE_URL = (
    "https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/GLOBAL/"
    "climatologies/1981-2010/bio/CHELSA_{var}_1981-2010_V.2.1.tif"
)


def _local_candidate_paths(chelsa_dir: Path, var: str) -> list[Path]:
    return [
        chelsa_dir / f"CHELSA_{var}_1981-2010_V.2.1.tif",
        chelsa_dir / f"CHELSA_{var}.tif",
        chelsa_dir / f"{var}.tif",
    ]


def _resolve_raster_path_or_url(chelsa_dir: Path, var: str) -> str:
    for p in _local_candidate_paths(chelsa_dir, var):
        if p.exists():
            return str(p)
    return CHELSA_BASE_URL.format(var=var)


def run(input_csv: Path = DATA_DIR / "sample_data_filtered.csv") -> pd.DataFrame:
    chelsa_dir = DATA_DIR / "chelsa"

    df = load_points(input_csv)
    valid = df["__valid_latlon"].to_numpy()
    lats = df["Latitude"].to_numpy(float)
    lons = df["Longitude"].to_numpy(float)

    out = pd.DataFrame(index=df.index)

    log_api_data(
        [
            "CHELSA BioClim: sampling bio1..bio19",
            "Dataset: CHELSA V2.1 baseline climatology 1981-2010",
            "URL pattern (if not using local files):",
            CHELSA_BASE_URL,
            f"Local dir override: {chelsa_dir}",
            "NOTE: For reproducibility, file SHA256 is logged when local rasters are used.",
        ],
        script_name=Path(__file__).name,
    )

    for var in BIOVARS:
        src = _resolve_raster_path_or_url(chelsa_dir, var)
        col = f"chelsa_{var}"
        vals = np.full(len(df), np.nan, dtype="float64")

        # Log provenance
        if src.startswith("http"):
            log_api_data([f"CHELSA {var}: remote={src}"], script_name=Path(__file__).name)
            raster_path = src  # rasterio can open remote URL
        else:
            p = Path(src)
            log_api_data(
                [
                    f"CHELSA {var}: local={p}",
                    f"CHELSA {var}: sha256={sha256_file(p)}",
                ],
                script_name=Path(__file__).name,
            )
            raster_path = str(p)

        # Sample
        vals[valid] = sample_raster_at_points(Path(raster_path) if not raster_path.startswith("http") else Path(raster_path), lons[valid], lats[valid])
        out[col] = vals

    # Save per-dataset output
    chelsa_dir.mkdir(parents=True, exist_ok=True)
    out_path = chelsa_dir / "chelsa_bioclim_points.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run()
