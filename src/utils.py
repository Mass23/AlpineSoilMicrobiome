#!/usr/bin/env python3
"""Shared utilities for metadata scripts.

All scripts in src/ should:
- read `data/sample_data_filtered.csv`
- use Latitude/Longitude in EPSG:4326
- write per-dataset CSV outputs under data/ (ignored by git)
- append provenance to data/api_data_log.txt

"""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
API_LOG = DATA_DIR / "api_data_log.txt"


def clamp_cpus(n: int) -> int:
    c = os.cpu_count() or 1
    return max(1, min(n, c))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_api_data(lines: list[str], script_name: Optional[str] = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    header = [
        "=" * 80,
        f"UTC timestamp: {timestamp}",
        f"Script: {script_name or ''}",
        f"Python: {sys.version.replace(os.linesep, ' ')}",
        f"Platform: {platform.platform()}",
    ]
    with API_LOG.open("a", encoding="utf-8") as f:
        for l in header + lines + [""]:
            f.write(l.rstrip() + "\n")


def load_points(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")

    df = df.copy()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    invalid = (
        df["Latitude"].isna()
        | df["Longitude"].isna()
        | (df["Latitude"] < -90)
        | (df["Latitude"] > 90)
        | (df["Longitude"] < -180)
        | (df["Longitude"] > 180)
    )
    df["__valid_latlon"] = ~invalid
    return df


def require_rasterio():
    try:
        import rasterio  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "rasterio is required. Create the conda env with python_topo_climate.yml and rerun."
        ) from e


def sample_raster_at_points(raster_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Sample band-1 at lon/lat points (EPSG:4326). Returns float64 with NaNs for nodata."""
    require_rasterio()
    import rasterio
    import rasterio.warp

    vals = np.full(shape=(len(lons),), fill_value=np.nan, dtype="float64")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise RuntimeError(f"Raster has no CRS: {raster_path}")

        if src.crs.to_string().upper() != "EPSG:4326":
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        else:
            xs, ys = lons.tolist(), lats.tolist()

        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                for v in src.sample([(x, y)]):
                    vv = float(v[0])
                    if src.nodata is not None and vv == float(src.nodata):
                        vals[i] = np.nan
                    else:
                        vals[i] = vv
            except Exception:
                vals[i] = np.nan

    return vals
