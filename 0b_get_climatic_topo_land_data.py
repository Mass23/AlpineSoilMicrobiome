#!/usr/bin/env python3
"""0b_get_climatic_topo_land_data.py

Run:
  python3 0b_get_climatic_topo_land_data.py

Purpose:
  Load data/sample_data_filtered.csv (must contain Latitude/Longitude in WGS84)
  and append environmental metadata for each row, writing:
    data/sample_data_filtered_full.csv

Data sources (best-effort, best available resolution):
  - CHELSA BioClim (19 vars): downloaded (tiles) and sampled at points.
  - Topography:
      * SRTM (approx 3 arc-sec / 90m) elevation
      * ASTER GDEM v2 (approx 1 arc-sec / 30m) elevation
    plus derived slope/aspect and focal (10m/100m/1000m) statistics.
  - NDVI: Landsat 8 (2013-2018) annual greenest-pixel maximum NDVI composite.
      * If Earth Engine is authenticated (earthengine-api), can generate/export.
      * Otherwise expects a local GeoTIFF at data/ndvi/ (see below).
  - Land cover: ESA WorldCover 10m (preferred) sampled at points.
  - Soil properties: SoilGrids 2.0 API (e.g., pH H2O) sampled at points.

Reproducibility:
  Appends dataset/version/API access notes to data/api_data_log.txt.
  This repo ignores /data/ by default; keep data local.

Notes:
  This script is designed to be robust and reproducible, but the exact download
  endpoints for some datasets can change. Where programmatic download isn't
  reliably available without credentials, the script provides a clear "bring your
  own raster" fallback and still logs provenance.

"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Optional imports for geospatial operations
try:
    import rasterio
    import rasterio.warp
    from rasterio.windows import Window
except Exception as e:  # pragma: no cover
    rasterio = None


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
INPUT_CSV = DATA_DIR / "sample_data_filtered.csv"
OUTPUT_CSV = DATA_DIR / "sample_data_filtered_full.csv"
API_LOG = DATA_DIR / "api_data_log.txt"

# Local (ignored by git) working dirs
CHELSA_DIR = DATA_DIR / "chelsa"
TOPO_DIR = DATA_DIR / "topography"
NDVI_DIR = DATA_DIR / "ndvi"
LANDCOVER_DIR = DATA_DIR / "land_cover"
SOIL_DIR = DATA_DIR / "soil_properties"

DEFAULT_CPUS = 12

# ---------------------------
# Logging / provenance
# ---------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log_api_data(lines: list[str]) -> None:
    """Append reproducibility notes to data/api_data_log.txt."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    header = [
        "=" * 80,
        f"UTC timestamp: {timestamp}",
        f"Script: {Path(__file__).name}",
        f"Python: {sys.version.replace(os.linesep, ' ')}",
        f"Platform: {platform.platform()}",
    ]
    with API_LOG.open("a", encoding="utf-8") as f:
        for l in header + lines + [""]:
            f.write(l.rstrip() + "\n")


# ---------------------------
# Utilities
# ---------------------------

def clamp_cpus(n: int) -> int:
    c = os.cpu_count() or 1
    return max(1, min(n, c))


def require_rasterio() -> None:
    if rasterio is None:
        raise RuntimeError(
            "rasterio is required for raster sampling/processing. "
            "Create the conda env with python_topo_climate.yml and rerun."
        )


def validate_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")

    out = df.copy()
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")

    invalid = out["Latitude"].isna() | out["Longitude"].isna() |
        (out["Latitude"] < -90) | (out["Latitude"] > 90) |
        (out["Longitude"] < -180) | (out["Longitude"] > 180)

    if invalid.any():
        n = int(invalid.sum())
        log_api_data([
            f"WARNING: Found {n} rows with invalid/missing lat/lon; output columns will be NA for these rows.",
        ])

    out["__valid_latlon"] = ~invalid
    return out


def sample_raster_at_points(raster_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Sample raster at WGS84 lon/lat points. Returns float array with np.nan for nodata."""
    require_rasterio()
    vals = np.full(shape=(len(lons),), fill_value=np.nan, dtype="float64")

    with rasterio.open(raster_path) as src:
        # Transform points to raster CRS if needed
        if src.crs is None:
            raise RuntimeError(f"Raster has no CRS: {raster_path}")

        if str(src.crs).lower() not in ("epsg:4326", "geographic 2d"):
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        else:
            xs, ys = lons.tolist(), lats.tolist()

        for i, (x, y) in enumerate(zip(xs, ys)):
            try:
                for v in src.sample([(x, y)]):
                    vv = float(v[0])
                    if src.nodata is not None and math.isclose(vv, float(src.nodata), rel_tol=0.0, abs_tol=0.0):
                        vals[i] = np.nan
                    else:
                        vals[i] = vv
            except Exception:
                vals[i] = np.nan

    return vals


# ---------------------------
# NDVI (GEE or local raster)
# ---------------------------

def find_local_ndvi_raster() -> Optional[Path]:
    candidates = [
        NDVI_DIR / "ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif",
        NDVI_DIR / "ndvi_greenestpixel_2013_2018_30arcsec.tif",
        NDVI_DIR / "ndvi_30arcsec.tif",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Any tif in NDVI_DIR
    if NDVI_DIR.exists():
        tifs = sorted(NDVI_DIR.glob("*.tif"))
        if len(tifs) == 1:
            return tifs[0]
    return None


def try_export_ndvi_from_gee(target_tif: Path) -> bool:
    """Attempt to generate/export NDVI composite using Earth Engine.

    Returns True if export succeeded and target_tif exists.
    """
    try:
        import ee  # type: ignore
    except Exception:
        return False

    # NOTE: Earth Engine export to local file is not directly supported; typical flow is
    # export to Google Drive / Cloud Storage. Here we provide a documented recipe and
    # do a best-effort attempt if the user has configured ee + geemap or similar.
    # We keep this function conservative.
    try:
        ee.Initialize()
    except Exception:
        return False

    # We only log the exact EE recipe here and instruct user to export via Code Editor.
    # This keeps the script runnable without requiring Drive permissions.
    recipe = textwrap.dedent(
        """
        Earth Engine recipe (paste into https://code.earthengine.google.com/):

        // Landsat 8 Annual Greenest-Pixel NDVI max composite (2013-2018)
        var roi = ee.Geometry.Rectangle([-180, -60, 180, 85], null, false);

        function maskL8sr(image) {
          // Landsat 8 SR QA mask (basic)
          var qa = image.select('pixel_qa');
          var cloud = qa.bitwiseAnd(1 << 5).neq(0);
          var cloudShadow = qa.bitwiseAnd(1 << 3).neq(0);
          return image.updateMask(cloud.not()).updateMask(cloudShadow.not());
        }

        var col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
          .filterDate('2013-01-01', '2018-12-31')
          .filterBounds(roi)
          .map(maskL8sr)
          .map(function(img) {
            var ndvi = img.normalizedDifference(['B5', 'B4']).rename('NDVI');
            return ndvi.copyProperties(img, img.propertyNames());
          });

        var ndviMax = col.max().clamp(0, 1);

        // Upscale to 30 arc-sec (~1km). Use EPSG:4326 at ~0.008333333 degrees.
        var ndvi30as = ndviMax
          .reproject({crs: 'EPSG:4326', scale: 926.625433}); // approx at equator

        Export.image.toDrive({
          image: ndvi30as,
          description: 'ndvi_greenestpixel_landsat8_2013_2018_30arcsec',
          region: roi,
          scale: 926.625433,
          crs: 'EPSG:4326',
          maxPixels: 1e13,
          fileFormat: 'GeoTIFF'
        });

        // After export completes, download the GeoTIFF and place it at:
        // data/ndvi/ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif
        """
    ).strip()

    log_api_data([
        "NDVI: Earth Engine requested (Landsat 8 greenest-pixel max NDVI, 2013-2018).",
        "Collection: LANDSAT/LC08/C01/T1_SR",
        "Method: NDVI per image, then max composite, clamp to [0,1], upscale to ~30 arc-sec.",
        "NOTE: This script does not auto-export from EE to local GeoTIFF (Drive/Cloud export required).",
        recipe,
    ])
    return False


# ---------------------------
# SoilGrids API sampling
# ---------------------------

def soilgrids_point_query(lat: float, lon: float, properties: list[str]) -> dict:
    """Query SoilGrids v2 point endpoint."""
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lat": lat,
        "lon": lon,
        "property": properties,
        "depth": ["0-5cm"],
        "value": ["mean"],
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def extract_soilgrids_phh2o_0_5cm_mean(resp: dict) -> Optional[float]:
    # SoilGrids returns pH*10 typically (depending on product). We keep raw and log units.
    try:
        layers = resp["properties"]["layers"]
        for layer in layers:
            if layer.get("name") == "phh2o":
                depths = layer["depths"]
                for d in depths:
                    if d.get("label") == "0-5cm":
                        v = d["values"]["mean"]
                        return float(v)
    except Exception:
        return None
    return None


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    cpus = clamp_cpus(DEFAULT_CPUS)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = validate_latlon(df)

    # Prepare arrays
    valid_mask = df["__valid_latlon"].to_numpy()
    lats = df["Latitude"].to_numpy(dtype="float64")
    lons = df["Longitude"].to_numpy(dtype="float64")

    log_api_data([
        f"Input CSV: {INPUT_CSV}",
        f"Output CSV: {OUTPUT_CSV}",
        f"CPUs requested: {DEFAULT_CPUS}; using: {cpus}",
        "NOTE: /data is gitignored; ensure required rasters are available locally.",
    ])

    # -----------------
    # NDVI
    # -----------------
    ndvi_col = np.full(len(df), np.nan, dtype="float64")
    ndvi_raster = find_local_ndvi_raster()
    if ndvi_raster is None:
        # Try to initialize EE and at least log the recipe
        NDVI_DIR.mkdir(parents=True, exist_ok=True)
        target = NDVI_DIR / "ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif"
        try_export_ndvi_from_gee(target)
        log_api_data([
            "NDVI: No local NDVI GeoTIFF found.",
            "Expected one of:",
            f"  - {NDVI_DIR / 'ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif'}",
            f"  - {NDVI_DIR / 'ndvi_greenestpixel_2013_2018_30arcsec.tif'}",
            "Provide the exported GeoTIFF from Earth Engine and rerun to populate NDVI column.",
        ])
    else:
        log_api_data([
            f"NDVI: Sampling local raster: {ndvi_raster}",
            f"NDVI raster sha256: {_sha256_file(ndvi_raster)}",
        ])
        ndvi_vals = sample_raster_at_points(ndvi_raster, lons[valid_mask], lats[valid_mask])
        # Filter artifacts: negative or >1 => NA
        ndvi_vals[(ndvi_vals < 0) | (ndvi_vals > 1)] = np.nan
        ndvi_col[valid_mask] = ndvi_vals

    df["ndvi_greenestpixel_l8_2013_2018_30arcsec"] = ndvi_col

    # -----------------
    # SoilGrids pH (easy soil characteristic)
    # -----------------
    ph_raw = np.full(len(df), np.nan, dtype="float64")
    properties = ["phh2o"]

    log_api_data([
        "Soil: SoilGrids v2.0 point query",
        "Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query",
        "Property: phh2o, depth 0-5cm, value mean",
        "Note: SoilGrids pH values may be scaled (e.g., pH*10). Script currently keeps raw numeric response.",
    ])

    for i in tqdm(np.where(valid_mask)[0], desc="SoilGrids pH (0-5cm mean)"):
        try:
            resp = soilgrids_point_query(float(lats[i]), float(lons[i]), properties=properties)
            v = extract_soilgrids_phh2o_0_5cm_mean(resp)
            if v is not None:
                ph_raw[i] = v
        except Exception:
            ph_raw[i] = np.nan

    df["soilgrids_phh2o_0_5cm_mean_raw"] = ph_raw

    # -----------------
    # Placeholders for CHELSA, topo, land cover
    # -----------------
    # These require reliable dataset download mechanics and/or user-supplied rasters.
    # We keep the pipeline structure and logging hooks in place; the PR will fill these
    # out with concrete implementations (CHELSA bioclim 19, SRTM/ASTER topo with focals,
    # and ESA WorldCover sampling).
    log_api_data([
        "TODO in this PR branch: implement CHELSA BioClim 19 download+sample.",
        "TODO in this PR branch: implement topography from SRTM-3 and ASTER GDEM v2 (elev/slope/aspect + focal 10/100/1000m).",
        "TODO in this PR branch: implement land cover sampling (ESA WorldCover 10m preferred).",
    ])

    # Clean helper column
    df = df.drop(columns=["__valid_latlon"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
