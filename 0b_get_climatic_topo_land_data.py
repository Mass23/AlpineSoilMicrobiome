#!/usr/bin/env python3
"""
0b_get_climatic_topo_land_data.py

Enriches data/sample_data_filtered.csv with environmental covariates:

  1. CHELSA BioClim v2.1  – 19 bioclimatic variables (bio1–bio19)
     ~30 arcsec (~1 km)  ·  climatology 1981–2010
     Remote access via VSICURL (no full-raster download required)

  2. Topography
     SRTM v4.1 (CGIAR-CSI, ~90 m)                 prefix: srtm_
     ASTER GDEM v3 (OpenTopography, ~30 m)         prefix: aster_   [optional]
     Per DEM: elevation, slope, aspect,
              focal mean/std at 10 m, 100 m, 1000 m scales

  3. NDVI  – local raster from Ecography doi:10.1111/ecog.05012
     Expected file: data/ndvi/<any *.tif or *.tiff>
     (place it manually – dataset is not programmatically downloadable)

  4. ESA WorldCover 2021 10 m  – land-cover class code & name
     Tiles accessed via VSICURL from public S3

  5. SoilGrids v2.0 REST API  – pH, bulk density, organic carbon, clay, sand
     at multiple depth horizons  (no authentication required)

Output:  data/sample_data_filtered_full.csv
Log:     data/api_data_log.txt   (dataset names, versions, URLs, hashes, timestamps)

Usage:
    python3 0b_get_climatic_topo_land_data.py [--n-cpus 12]

Optional environment variables:
    OPENTOPOGRAPHY_API_KEY   Free key from https://portal.opentopography.org
                             Enables SRTM 30 m (SRTMGL1) and ASTER GDEM v3
                             downloads.  Without it the script uses CGIAR-CSI
                             SRTM v4.1 (~90 m) for SRTM and skips ASTER.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.transform import rowcol as rasterio_rowcol
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

N_CPUS_DEFAULT: int = 12
N_CPUS: int = min(N_CPUS_DEFAULT, os.cpu_count() or 1)

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR   = REPO_ROOT / "data"
CHELSA_DIR = DATA_DIR / "chelsa"
TOPO_DIR   = DATA_DIR / "topography"
NDVI_DIR   = DATA_DIR / "ndvi"
LC_DIR     = DATA_DIR / "land_cover"
SOIL_DIR   = DATA_DIR / "soil_properties"

INPUT_CSV  = DATA_DIR / "sample_data_filtered.csv"
OUTPUT_CSV = DATA_DIR / "sample_data_filtered_full.csv"
LOG_FILE   = DATA_DIR / "api_data_log.txt"

# Focal scales in metres
FOCAL_SCALES_M: List[int] = [10, 100, 1000]

# ── CHELSA V2.1 ───────────────────────────────────────────────────────────────
CHELSA_BASE    = (
    "https://os.zhdk.cloud.switch.ch/envicloud/chelsa/chelsa_V2/"
    "GLOBAL/climatologies/1981-2010/bio"
)
CHELSA_FNAME   = "CHELSA_bio{n}_1981-2010_V.2.1.tif"
CHELSA_VERSION = "V.2.1 (climatology 1981–2010)"

# ── OpenTopography REST API ───────────────────────────────────────────────────
OPENTOPO_URL     = "https://portal.opentopography.org/API/globaldem"
OPENTOPO_API_KEY = os.environ.get("OPENTOPOGRAPHY_API_KEY", "")

# ── CGIAR-CSI SRTM v4.1 (no authentication, fallback) ────────────────────────
CGIAR_BASE    = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF"
CGIAR_VERSION = "SRTM v4.1 (CGIAR-CSI)"

# ── SoilGrids v2.0 REST API ───────────────────────────────────────────────────
SOILGRIDS_URL     = "https://rest.isric.org/soilgrids/v2.0/properties/query"
SOILGRIDS_VERSION = "v2.0"

# Properties and depths to fetch
SOILGRIDS_PROPERTIES: Dict[str, List[str]] = {
    "phh2o": ["0-5cm", "5-15cm", "15-30cm"],
    "bdod":  ["0-5cm", "5-15cm"],
    "soc":   ["0-5cm", "5-15cm"],
    "clay":  ["0-5cm"],
    "sand":  ["0-5cm"],
}
# Raw SoilGrids values need scaling to standard units
SOILGRIDS_SCALE: Dict[str, float] = {
    "phh2o": 0.1,   # stored as pH × 10
    "bdod":  0.01,  # cg/cm³  →  g/cm³
    "soc":   0.1,   # dg/kg   →  g/kg
    "clay":  0.1,   # g/kg (stored × 10)
    "sand":  0.1,
}

# ── ESA WorldCover 2021 v200 ──────────────────────────────────────────────────
WORLDCOVER_BASE    = "https://esa-worldcover.s3.amazonaws.com/v200/2021/map"
WORLDCOVER_VERSION = "ESA WorldCover 2021 v200"
WORLDCOVER_CLASSES: Dict[int, str] = {
    10:  "Tree cover",
    20:  "Shrubland",
    30:  "Grassland",
    40:  "Cropland",
    50:  "Built-up",
    60:  "Bare / sparse vegetation",
    70:  "Snow and ice",
    80:  "Permanent water bodies",
    90:  "Herbaceous wetland",
    95:  "Mangroves",
    100: "Moss and lichen",
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Provenance utilities
# ──────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def append_log(entries: List[Dict[str, Any]]) -> None:
    """Append provenance entries to api_data_log.txt."""
    import importlib

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )
    lib_versions: Dict[str, str] = {}
    for lib in ("numpy", "pandas", "rasterio", "scipy", "requests"):
        try:
            lib_versions[lib] = importlib.import_module(lib).__version__
        except Exception:
            lib_versions[lib] = "unknown"

    with open(LOG_FILE, "a") as fh:
        fh.write(f"\n{'=' * 72}\n")
        fh.write(f"Access timestamp : {timestamp}\n")
        fh.write(f"Python version   : {py_ver}\n")
        fh.write(f"Library versions : {json.dumps(lib_versions)}\n")
        for e in entries:
            fh.write(f"--- Dataset: {e.get('dataset', 'unknown')}\n")
            for k, v in e.items():
                if k != "dataset":
                    fh.write(f"    {k}: {v}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Directory setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_directories() -> None:
    """Create required data subdirectories (and .gitkeep files)."""
    for d in (CHELSA_DIR, TOPO_DIR, NDVI_DIR, LC_DIR, SOIL_DIR):
        d.mkdir(parents=True, exist_ok=True)
        gk = d / ".gitkeep"
        if not gk.exists():
            gk.touch()
    logger.info("Data subdirectories ready.")


# ──────────────────────────────────────────────────────────────────────────────
# Input loading
# ──────────────────────────────────────────────────────────────────────────────

def load_input() -> pd.DataFrame:
    """Load and validate input CSV.  Marks invalid rows with _valid=False."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"Loaded {len(df)} rows from {INPUT_CSV}")

    for col in ("Latitude", "Longitude"):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from input CSV.")

    df["Latitude"]  = pd.to_numeric(df["Latitude"],  errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

    valid = (
        df["Latitude"].notna()  & df["Longitude"].notna()
        & df["Latitude"].between(-90,  90)
        & df["Longitude"].between(-180, 180)
    )
    n_bad = (~valid).sum()
    if n_bad:
        logger.warning(
            f"{n_bad} row(s) have missing/invalid Lat/Lon — "
            "these rows will have NaN for all environmental columns."
        )
    df["_valid"] = valid
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Raster utilities
# ──────────────────────────────────────────────────────────────────────────────

def sample_raster_at_coords(
    src: rasterio.DatasetReader,
    coords: List[Tuple[float, float]],
    band: int = 1,
) -> np.ndarray:
    """
    Sample a rasterio dataset at (lon, lat) pairs.
    Returns float array; nodata pixels become NaN.
    """
    nodata = src.nodata
    out: List[float] = []
    for val in src.sample(coords, indexes=band):
        v = float(val[0])
        if nodata is not None and v == float(nodata):
            out.append(np.nan)
        else:
            out.append(v)
    return np.array(out, dtype=np.float64)


def array_sample(
    arr: np.ndarray,
    transform: Any,
    lons: np.ndarray,
    lats: np.ndarray,
) -> np.ndarray:
    """
    Index a 2-D array at geographic coordinates using a rasterio Affine transform.
    Out-of-bounds coordinates return NaN.
    """
    nrows, ncols = arr.shape
    rows, cols = rasterio_rowcol(transform, lons, lats)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    valid = (rows >= 0) & (rows < nrows) & (cols >= 0) & (cols < ncols)
    out = np.full(len(lons), np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = arr[rows[valid], cols[valid]].astype(np.float64)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CHELSA BioClim v2.1
# ──────────────────────────────────────────────────────────────────────────────

def extract_chelsa(
    coords: List[Tuple[float, float]],
    n_cpus: int,
) -> pd.DataFrame:
    """
    Extract CHELSA BioClim v2.1 bio1–bio19 at coordinate list.

    Uses VSICURL so the full (~2.2 GB) GeoTIFF is never downloaded; only the
    compressed tiles covering the query points are fetched via HTTP range
    requests.  Falls back to a local cached download if VSICURL fails.
    """
    logger.info("Extracting CHELSA BioClim v2.1 (bio1–bio19)…")
    records: Dict[str, List[float]] = {}
    log_entries: List[Dict[str, Any]] = []

    for n in range(1, 20):
        col   = f"chelsa_bio{n}"
        fname = CHELSA_FNAME.format(n=n)
        url   = f"{CHELSA_BASE}/{fname}"
        vsurl = f"/vsicurl/{url}"
        local = CHELSA_DIR / fname

        try:
            # Prefer VSICURL (no download)
            with rasterio.open(vsurl) as src:
                vals = sample_raster_at_coords(src, coords)
            logger.info(f"  bio{n:2d}: sampled via VSICURL")
        except Exception:
            # Fall back: download to local cache
            if not local.exists():
                logger.info(f"  bio{n:2d}: VSICURL unavailable, downloading {fname}…")
                try:
                    r = requests.get(url, stream=True, timeout=600)
                    r.raise_for_status()
                    with open(local, "wb") as fh:
                        for chunk in r.iter_content(65536):
                            fh.write(chunk)
                    log_entries.append({
                        "dataset": f"CHELSA BioClim bio{n} (downloaded)",
                        "version": CHELSA_VERSION,
                        "url": url,
                        "sha256": _sha256(local),
                    })
                except Exception as exc2:
                    logger.error(f"  bio{n}: download failed: {exc2}")
                    records[col] = [np.nan] * len(coords)
                    continue
            try:
                with rasterio.open(local) as src:
                    vals = sample_raster_at_coords(src, coords)
                # Delete after use (keep only hash in log)
                local.unlink(missing_ok=True)
                logger.info(f"  bio{n:2d}: sampled from local cache (deleted)")
            except Exception as exc3:
                logger.error(f"  bio{n}: local sampling failed: {exc3}")
                records[col] = [np.nan] * len(coords)
                continue

        records[col] = vals.tolist()
        log_entries.append({
            "dataset": f"CHELSA BioClim bio{n}",
            "version": CHELSA_VERSION,
            "url": url,
        })

    append_log(log_entries)
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# DEM / Topography helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pixel_size_m(src: rasterio.DatasetReader) -> Tuple[float, float]:
    """Approximate pixel size in metres (WGS-84 geographic CRS assumed)."""
    crs = src.crs
    if crs and crs.is_geographic:
        lat_c = (src.bounds.top + src.bounds.bottom) / 2.0
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_c))
        px_lon = abs(src.transform.a)
        px_lat = abs(src.transform.e)
        return px_lon * m_per_deg_lon, px_lat * m_per_deg_lat
    return abs(src.transform.a), abs(src.transform.e)


def _focal_window(scale_m: float, res_m: float) -> int:
    """Odd window size (≥ 1) covering the requested radius in metres."""
    half = max(0, round(scale_m / res_m))
    return 2 * half + 1


def compute_dem_layers(
    dem: np.ndarray,
    nodata: Optional[float],
    res_x_m: float,
    res_y_m: float,
) -> Dict[str, np.ndarray]:
    """
    Compute elevation, slope, aspect, and focal statistics from a DEM array.

    Focal mean and std are computed for each scale in FOCAL_SCALES_M using a
    moving-window of the appropriate pixel radius:
      window_pixels = 2 * round(scale_m / pixel_size_m) + 1

    Returns a dict of {layer_name: 2D array}.
    """
    data = dem.astype(np.float64)
    if nodata is not None:
        data[dem == nodata] = np.nan

    nan_mask = np.isnan(data)
    layers: Dict[str, np.ndarray] = {"elev": data.copy()}

    # Slope and aspect via central finite differences
    # np.gradient respects the physical spacing (res_y_m, res_x_m)
    gy, gx = np.gradient(np.where(nan_mask, 0.0, data), res_y_m, res_x_m)
    slope = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    # Geographic convention: North = 0°, clockwise
    aspect = np.degrees(np.arctan2(-gy, gx))
    aspect = (90.0 - aspect) % 360.0
    slope[nan_mask]  = np.nan
    aspect[nan_mask] = np.nan
    layers["slope"]  = slope
    layers["aspect"] = aspect

    # Focal statistics per scale
    d  = np.where(nan_mask, 0.0, data)
    m  = (~nan_mask).astype(np.float64)
    d2 = d * d

    for scale_m in FOCAL_SCALES_M:
        win_x = _focal_window(scale_m, res_x_m)
        win_y = _focal_window(scale_m, res_y_m)
        size  = (win_y, win_x)

        # Mask-aware mean:  sum(val * valid) / sum(valid)
        fsum  = uniform_filter(d * m, size=size)
        fcnt  = uniform_filter(m,     size=size)
        fsum2 = uniform_filter(d2 * m, size=size)

        with np.errstate(invalid="ignore", divide="ignore"):
            fmean = np.where(fcnt > 0, fsum / fcnt, np.nan)
            fvar  = fsum2 / np.where(fcnt > 0, fcnt, np.nan) - fmean**2
            fstd  = np.sqrt(np.where(fvar > 0, fvar, 0.0))

        fmean[nan_mask] = np.nan
        fstd[nan_mask]  = np.nan

        tag = f"{scale_m}m"
        layers[f"focal_mean_{tag}"] = fmean
        layers[f"focal_std_{tag}"]  = fstd

    return layers


def sample_dem_layers(
    raster_path: Path,
    lons: np.ndarray,
    lats: np.ndarray,
    prefix: str,
) -> pd.DataFrame:
    """
    Open a DEM GeoTIFF, compute topographic layers, and sample at coordinates.
    Returns a DataFrame with columns named {prefix}{layer}.
    """
    with rasterio.open(raster_path) as src:
        dem    = src.read(1)
        nodata = src.nodata
        tr     = src.transform
        rx_m, ry_m = _pixel_size_m(src)

    dem_layers = compute_dem_layers(dem, nodata, rx_m, ry_m)

    records: Dict[str, np.ndarray] = {}
    for layer_name, arr in dem_layers.items():
        col = f"{prefix}{layer_name}"
        records[col] = array_sample(arr, tr, lons, lats)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# SRTM v4.1 via CGIAR-CSI (no authentication)
# ──────────────────────────────────────────────────────────────────────────────

def _cgiar_tile(lat: float, lon: float) -> Tuple[int, int]:
    """
    Return (col, row) of the CGIAR-CSI SRTM v4.1 tile that contains (lat, lon).

    Tiles are 5° × 5° (col 1–72, row 1–24), covering 60°N to 60°S.
    """
    col = int(math.floor((lon + 180.0) / 5.0)) + 1
    row = int(math.floor((60.0 - lat)  / 5.0)) + 1
    col = max(1, min(72, col))
    row = max(1, min(24, row))
    return col, row


def download_srtm_cgiar(
    lats: np.ndarray,
    lons: np.ndarray,
    out_dir: Path,
) -> Optional[Path]:
    """
    Download the CGIAR-CSI SRTM v4.1 tiles needed to cover all points, merge
    them with rasterio if multiple tiles are needed, and return the path to
    the merged (or single) GeoTIFF.

    Tiles are cached in out_dir.  Returns None on failure.
    """
    tile_ids = {_cgiar_tile(lat, lon) for lat, lon in zip(lats, lons)}
    logger.info(f"SRTM CGIAR: fetching {len(tile_ids)} tile(s): {sorted(tile_ids)}")

    tile_paths: List[Path] = []
    log_entries: List[Dict[str, Any]] = []

    for col, row in sorted(tile_ids):
        fname    = f"srtm_{col:02d}_{row:02d}.zip"
        url      = f"{CGIAR_BASE}/{fname}"
        zip_path = out_dir / fname
        tif_path = out_dir / fname.replace(".zip", ".tif")

        if tif_path.exists():
            logger.info(f"  Using cached tile {tif_path.name}")
            tile_paths.append(tif_path)
            continue

        logger.info(f"  Downloading {fname} …")
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in r.iter_content(65536):
                    fh.write(chunk)
            sha = _sha256(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
                if not tif_names:
                    logger.error(f"  No TIF found inside {fname}")
                    zip_path.unlink(missing_ok=True)
                    continue
                zf.extract(tif_names[0], out_dir)
                extracted = out_dir / tif_names[0]
                if extracted != tif_path:
                    extracted.rename(tif_path)
            zip_path.unlink(missing_ok=True)
            log_entries.append({
                "dataset": f"SRTM tile srtm_{col:02d}_{row:02d}",
                "version": CGIAR_VERSION,
                "url": url,
                "sha256_zip": sha,
            })
            tile_paths.append(tif_path)
        except requests.HTTPError as http_err:
            logger.warning(f"  Tile srtm_{col:02d}_{row:02d} not available ({http_err}) — skipping")
            zip_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.error(f"  Tile download failed: {exc}")
            zip_path.unlink(missing_ok=True)

    if log_entries:
        append_log(log_entries)

    if not tile_paths:
        logger.error("No SRTM tiles downloaded successfully.")
        return None

    if len(tile_paths) == 1:
        return tile_paths[0]

    # Merge multiple tiles with rasterio.merge
    merged_path = out_dir / "srtm_merged_tmp.tif"
    try:
        from rasterio.merge import merge as rio_merge

        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, out_trans = rio_merge(datasets)
        out_meta = datasets[0].meta.copy()
        out_meta.update({
            "driver":    "GTiff",
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": out_trans,
        })
        with rasterio.open(merged_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        for ds in datasets:
            ds.close()
        logger.info(f"  Merged {len(tile_paths)} SRTM tiles → {merged_path.name}")
        return merged_path
    except Exception as exc:
        logger.error(f"SRTM merge failed: {exc} — using first tile only")
        return tile_paths[0]


# ──────────────────────────────────────────────────────────────────────────────
# OpenTopography REST API (SRTM 30 m + ASTER GDEM v3)
# ──────────────────────────────────────────────────────────────────────────────

def download_opentopo(
    south: float,
    north: float,
    west: float,
    east: float,
    demtype: str,
    out_path: Path,
    api_key: str,
) -> bool:
    """
    Download a DEM bounding-box raster from OpenTopography as GeoTIFF.

    demtype options: SRTMGL1 (30 m), SRTMGL3 (90 m), ASTGTM (ASTER GDEM v3)
    Returns True on success, False otherwise.
    """
    params = {
        "demtype":      demtype,
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": "GTiff",
        "API_Key":      api_key,
    }
    logger.info(
        f"OpenTopography {demtype}: bbox "
        f"S{south:.3f} N{north:.3f} W{west:.3f} E{east:.3f} …"
    )
    try:
        r = requests.get(OPENTOPO_URL, params=params, stream=True, timeout=600)
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(65536):
                fh.write(chunk)
        logger.info(f"  Saved {demtype} raster → {out_path.name}")
        return True
    except Exception as exc:
        logger.error(f"  OpenTopography {demtype} failed: {exc}")
        out_path.unlink(missing_ok=True)
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Topography extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_topography(
    coords: List[Tuple[float, float]],
    n_cpus: int,
) -> pd.DataFrame:
    """
    Extract topographic variables (elevation, slope, aspect, focal mean/std)
    from SRTM and, optionally, ASTER GDEM.

    SRTM source:
      - With OPENTOPOGRAPHY_API_KEY: SRTMGL1 (~30 m) via OpenTopography
      - Without key            : CGIAR-CSI SRTM v4.1 (~90 m) tiles

    ASTER GDEM v3 (~30 m):
      - Requires OPENTOPOGRAPHY_API_KEY; skipped otherwise.

    All points are processed together using a single bounding-box raster
    (+ 0.1° buffer for boundary focal windows).  Merged/temporary rasters
    are deleted after sampling; per-tile CGIAR files are kept as local cache.
    """
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])

    buf   = 0.1          # degrees (~11 km) — ensures 1 km focal windows work at edges
    south = float(lats.min()) - buf
    north = float(lats.max()) + buf
    west  = float(lons.min()) - buf
    east  = float(lons.max()) + buf

    result_dfs: List[pd.DataFrame] = []

    # ── SRTM ──────────────────────────────────────────────────────────────────
    srtm_path: Optional[Path] = None
    srtm_version = CGIAR_VERSION
    is_temp_srtm = False

    if OPENTOPO_API_KEY:
        srtm_path    = TOPO_DIR / "srtm30m_aoi.tif"
        srtm_version = "SRTMGL1 (30 m) via OpenTopography"
        is_temp_srtm = False  # keep for re-use; user can delete
        if not srtm_path.exists():
            ok = download_opentopo(south, north, west, east, "SRTMGL1",
                                   srtm_path, OPENTOPO_API_KEY)
            if not ok:
                logger.warning(
                    "OpenTopography SRTM 30 m failed; "
                    "falling back to CGIAR-CSI SRTM v4.1 (~90 m)."
                )
                srtm_path    = download_srtm_cgiar(lats, lons, TOPO_DIR)
                srtm_version = CGIAR_VERSION
                is_temp_srtm = (
                    srtm_path is not None
                    and srtm_path.name == "srtm_merged_tmp.tif"
                )
        else:
            logger.info("Using cached SRTM 30 m raster.")
    else:
        logger.info("No OpenTopography API key — using CGIAR-CSI SRTM v4.1 (~90 m).")
        srtm_path = download_srtm_cgiar(lats, lons, TOPO_DIR)
        # merged file is temporary; individual tile TIFs are kept as cache
        is_temp_srtm = srtm_path is not None and srtm_path.name == "srtm_merged_tmp.tif"

    if srtm_path and srtm_path.exists():
        logger.info(f"Computing SRTM topographic layers from {srtm_path.name}…")
        srtm_df = sample_dem_layers(srtm_path, lons, lats, prefix="srtm_")
        sha = _sha256(srtm_path)
        append_log([{
            "dataset": "SRTM",
            "version": srtm_version,
            "file":    srtm_path.name,
            "sha256":  sha,
        }])
        result_dfs.append(srtm_df)
        if is_temp_srtm:
            srtm_path.unlink(missing_ok=True)
            logger.info("Removed temporary merged SRTM raster.")
    else:
        logger.warning("SRTM extraction skipped (no raster available).")

    # ── ASTER GDEM v3 ─────────────────────────────────────────────────────────
    if OPENTOPO_API_KEY:
        aster_path = TOPO_DIR / "aster_gdem3_aoi.tif"
        if not aster_path.exists():
            ok = download_opentopo(south, north, west, east, "ASTGTM",
                                   aster_path, OPENTOPO_API_KEY)
        else:
            logger.info("Using cached ASTER GDEM v3 raster.")
            ok = True

        if ok and aster_path.exists():
            logger.info("Computing ASTER GDEM v3 topographic layers…")
            aster_df = sample_dem_layers(aster_path, lons, lats, prefix="aster_")
            sha = _sha256(aster_path)
            append_log([{
                "dataset": "ASTER GDEM v3",
                "version": "ASTGTM via OpenTopography",
                "file":    aster_path.name,
                "sha256":  sha,
            }])
            result_dfs.append(aster_df)
        else:
            logger.warning("ASTER GDEM v3 extraction skipped.")
    else:
        logger.warning(
            "ASTER GDEM v3 skipped — set OPENTOPOGRAPHY_API_KEY (free at "
            "https://portal.opentopography.org) and re-run to enable."
        )

    if result_dfs:
        return pd.concat(result_dfs, axis=1)
    return pd.DataFrame(index=range(len(coords)))


# ──────────────────────────────────────────────────────────────────────────────
# NDVI (local raster)
# ──────────────────────────────────────────────────────────────────────────────

def extract_ndvi(coords: List[Tuple[float, float]]) -> pd.Series:
    """
    Sample NDVI from a local raster placed in data/ndvi/.

    The raster should be the global NDVI product from:
      Moeslund et al. (2022) Ecography doi:10.1111/ecog.05012

    Place any *.tif / *.tiff file in data/ndvi/ before running.
    If no file is found the column is set to NaN.
    """
    tif_files = list(NDVI_DIR.glob("*.tif")) + list(NDVI_DIR.glob("*.tiff"))
    if not tif_files:
        logger.warning(
            f"No NDVI raster found in {NDVI_DIR}. "
            "Download/place the GeoTIFF from doi:10.1111/ecog.05012 there "
            "and re-run. Column 'ndvi' will be NaN for this run."
        )
        return pd.Series([np.nan] * len(coords), name="ndvi")

    ndvi_path = tif_files[0]
    if len(tif_files) > 1:
        logger.warning(
            f"Multiple NDVI rasters found in {NDVI_DIR}; using {ndvi_path.name}"
        )

    logger.info(f"Sampling NDVI from {ndvi_path.name}…")
    with rasterio.open(ndvi_path) as src:
        vals = sample_raster_at_coords(src, coords)

    append_log([{
        "dataset": "NDVI (Ecography doi:10.1111/ecog.05012)",
        "version": "local raster",
        "file":    ndvi_path.name,
        "sha256":  _sha256(ndvi_path),
    }])
    return pd.Series(vals, name="ndvi")


# ──────────────────────────────────────────────────────────────────────────────
# ESA WorldCover 2021 (land cover)
# ──────────────────────────────────────────────────────────────────────────────

def _worldcover_tile_url(lat: float, lon: float) -> Tuple[str, str]:
    """
    Return (tile_name, HTTP URL) for the ESA WorldCover 2021 v200 tile that
    contains the given point.

    Tiles are 3° × 3°, named by their lower-left corner:
      {N|S}{abs(lat):02d}{E|W}{abs(lon):03d}
    e.g. N45E006 covers lat 45–48, lon 6–9.
    """
    tile_lat = int(math.floor(lat / 3.0)) * 3
    tile_lon = int(math.floor(lon / 3.0)) * 3
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    tile  = f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"
    fname = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    url   = f"{WORLDCOVER_BASE}/{fname}"
    return tile, url


def _sample_worldcover_tile(
    url: str,
    tile_coords: List[Tuple[float, float]],
    tile_indices: List[int],
) -> Dict[int, int]:
    """
    Sample one WorldCover tile (VSICURL) for the given points.
    Returns {original_index: class_code}.  -1 indicates nodata/failure.
    """
    vsurl  = f"/vsicurl/{url}"
    result: Dict[int, int] = {}
    try:
        with rasterio.open(vsurl) as src:
            nodata = src.nodata
            for idx, val in zip(tile_indices, src.sample(tile_coords)):
                v = int(val[0])
                result[idx] = -1 if (nodata is not None and v == int(nodata)) else v
    except Exception as exc:
        logger.debug(f"WorldCover tile {url} failed: {exc}")
        for idx in tile_indices:
            result[idx] = -1
    return result


def extract_landcover(
    coords: List[Tuple[float, float]],
    n_cpus: int,
) -> pd.DataFrame:
    """
    Extract ESA WorldCover 2021 10 m land-cover class for all coordinates.

    WorldCover tiles are read via VSICURL from the public ESA S3 bucket;
    no full-tile downloads are stored.  Requests for distinct tiles run in
    parallel (up to n_cpus threads).
    """
    logger.info("Extracting ESA WorldCover 2021 land-cover…")

    # Group point indices by tile
    tile_groups: Dict[str, Tuple[str, List[int], List[Tuple[float, float]]]] = {}
    for i, (lon, lat) in enumerate(coords):
        tile, url = _worldcover_tile_url(lat, lon)
        if tile not in tile_groups:
            tile_groups[tile] = (url, [], [])
        tile_groups[tile][1].append(i)
        tile_groups[tile][2].append((lon, lat))

    class_vals = np.full(len(coords), -1, dtype=int)
    log_entries: List[Dict[str, Any]] = []

    def _task(item: Tuple[str, Tuple[str, List[int], List[Tuple[float, float]]]]):
        tile, (url, indices, tc) = item
        return tile, url, _sample_worldcover_tile(url, tc, indices)

    with ThreadPoolExecutor(max_workers=n_cpus) as exe:
        futures = {exe.submit(_task, item): item[0] for item in tile_groups.items()}
        for future in as_completed(futures):
            tile = futures[future]
            try:
                tile_name, url, res = future.result()
                for idx, v in res.items():
                    class_vals[idx] = v
                log_entries.append({
                    "dataset": f"ESA WorldCover 2021 tile {tile_name}",
                    "version": WORLDCOVER_VERSION,
                    "url":     url,
                })
            except Exception as exc:
                logger.error(f"WorldCover tile {tile} exception: {exc}")

    append_log(log_entries)

    classes    = np.where(class_vals == -1, np.nan, class_vals.astype(float))
    class_name = [
        WORLDCOVER_CLASSES.get(int(v), "Unknown") if not np.isnan(v) else np.nan
        for v in classes
    ]
    return pd.DataFrame({
        "worldcover_class": classes,
        "worldcover_name":  class_name,
    })


# ──────────────────────────────────────────────────────────────────────────────
# SoilGrids v2.0
# ──────────────────────────────────────────────────────────────────────────────

def _query_soilgrids_point(lat: float, lon: float) -> Dict[str, float]:
    """
    Query SoilGrids v2.0 REST API for a single point.

    Returns a dict of {column_name: scaled_value}.
    No authentication required; rate-limited to be polite.
    """
    props  = list(SOILGRIDS_PROPERTIES.keys())
    depths = sorted({d for dl in SOILGRIDS_PROPERTIES.values() for d in dl})

    params: Any = {
        "lon":      lon,
        "lat":      lat,
        "property": props,
        "depth":    depths,
        "value":    "mean",
    }
    result: Dict[str, float] = {}
    try:
        r = requests.get(
            SOILGRIDS_URL, params=params, timeout=60,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        for layer in data.get("properties", {}).get("layers", []):
            prop  = layer.get("name", "")
            scale = SOILGRIDS_SCALE.get(prop, 1.0)
            for depth_info in layer.get("depths", []):
                depth_label = depth_info.get("label", "").replace(" ", "")
                raw = depth_info.get("values", {}).get("mean")
                if raw is not None:
                    col = f"soilgrids_{prop}_{depth_label}_mean"
                    result[col] = float(raw) * scale
    except Exception as exc:
        logger.debug(f"SoilGrids query failed for ({lat:.4f}, {lon:.4f}): {exc}")
    return result


def extract_soilgrids(
    coords: List[Tuple[float, float]],
    n_cpus: int,
) -> pd.DataFrame:
    """
    Query SoilGrids v2.0 REST API for all coordinate pairs in parallel.

    Uses a thread pool (I/O-bound) with polite 50 ms inter-request delay.
    Responses are cached conceptually via log entries; results are returned
    in-memory and written to the output CSV.
    """
    logger.info(
        f"Querying SoilGrids v2.0 for {len(coords)} points "
        f"({n_cpus} parallel workers)…"
    )
    results: List[Optional[Dict[str, float]]] = [None] * len(coords)
    all_cols: set = set()

    def _query(i_coord: Tuple[int, Tuple[float, float]]):
        i, (lon, lat) = i_coord
        time.sleep(0.05)  # polite rate limiting
        return i, _query_soilgrids_point(lat, lon)

    with ThreadPoolExecutor(max_workers=n_cpus) as exe:
        futures = {
            exe.submit(_query, (i, c)): i
            for i, c in enumerate(coords)
        }
        done = 0
        for future in as_completed(futures):
            i, res = future.result()
            results[i] = res
            all_cols.update(res.keys())
            done += 1
            if done % 100 == 0 or done == len(coords):
                logger.info(f"  SoilGrids: {done}/{len(coords)} queries complete")

    all_cols_sorted = sorted(all_cols)
    rows = [
        {c: (r.get(c, np.nan) if r else np.nan) for c in all_cols_sorted}
        for r in results
    ]

    append_log([{
        "dataset":    "SoilGrids",
        "version":    SOILGRIDS_VERSION,
        "endpoint":   SOILGRIDS_URL,
        "properties": str(list(SOILGRIDS_PROPERTIES.keys())),
        "depths":     str(sorted({d for dl in SOILGRIDS_PROPERTIES.values() for d in dl})),
    }])

    return pd.DataFrame(rows, columns=all_cols_sorted)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main(n_cpus: int) -> None:
    logger.info("=" * 60)
    logger.info("0b_get_climatic_topo_land_data.py  –  starting")
    logger.info(f"  n_cpus     : {n_cpus}")
    logger.info(f"  input CSV  : {INPUT_CSV}")
    logger.info(f"  output CSV : {OUTPUT_CSV}")
    logger.info("=" * 60)

    setup_directories()

    df         = load_input()
    valid_mask = df["_valid"].values
    n_valid    = int(valid_mask.sum())
    logger.info(f"Processing {n_valid} valid coordinate pairs.")

    valid_df = df[valid_mask].reset_index(drop=False)          # keep orig index
    orig_idx = valid_df["index"].values                        # original row indices
    valid_df = valid_df.drop(columns=["index"])

    coords: List[Tuple[float, float]] = list(
        zip(valid_df["Longitude"].values, valid_df["Latitude"].values)
    )

    new_frames: List[pd.DataFrame] = []

    # 1. CHELSA BioClim
    try:
        new_frames.append(extract_chelsa(coords, n_cpus))
    except Exception as exc:
        logger.error(f"CHELSA extraction failed: {exc}")

    # 2. Topography
    try:
        new_frames.append(extract_topography(coords, n_cpus))
    except Exception as exc:
        logger.error(f"Topography extraction failed: {exc}")

    # 3. NDVI
    try:
        new_frames.append(extract_ndvi(coords).to_frame())
    except Exception as exc:
        logger.error(f"NDVI extraction failed: {exc}")

    # 4. Land cover
    try:
        new_frames.append(extract_landcover(coords, n_cpus))
    except Exception as exc:
        logger.error(f"Land-cover extraction failed: {exc}")

    # 5. SoilGrids
    try:
        new_frames.append(extract_soilgrids(coords, n_cpus))
    except Exception as exc:
        logger.error(f"SoilGrids extraction failed: {exc}")

    # Concatenate new columns aligned to valid rows
    if new_frames:
        new_cols = pd.concat(new_frames, axis=1)
    else:
        new_cols = pd.DataFrame(index=range(n_valid))

    # Build output: all original rows + new columns (NaN for invalid rows)
    out_df = df.drop(columns=["_valid"]).copy()

    if new_cols.columns.any():
        # Reindex new columns to cover all original rows (NaN for invalid ones)
        new_cols_full = new_cols.copy()
        new_cols_full.index = orig_idx
        new_cols_full = new_cols_full.reindex(out_df.index)

        # String/object columns: fill with None (not float NaN) for correct dtype
        for col in new_cols_full.columns:
            if pd.api.types.is_object_dtype(new_cols_full[col]):
                new_cols_full[col] = new_cols_full[col].where(
                    new_cols_full[col].notna(), other=None
                )
        out_df = pd.concat([out_df, new_cols_full], axis=1)

    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info(
        f"✓ Output saved → {OUTPUT_CSV}  "
        f"({len(out_df)} rows × {len(out_df.columns)} columns)"
    )

    # Final provenance entry
    append_log([{
        "dataset":          "Run summary",
        "input_file":       str(INPUT_CSV),
        "output_file":      str(OUTPUT_CSV),
        "n_rows_input":     len(df),
        "n_rows_valid":     n_valid,
        "n_columns_output": len(out_df.columns),
        "new_columns":      str(list(new_cols.columns)),
    }])


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Enrich sample coordinates with CHELSA bioclimatic variables, "
            "topography (SRTM/ASTER), NDVI, ESA WorldCover land cover, "
            "and SoilGrids soil properties."
        )
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=N_CPUS,
        help=(
            f"Number of parallel workers for API queries "
            f"(default: {N_CPUS}, max: {os.cpu_count()})."
        ),
    )
    args   = parser.parse_args()
    n_cpus = min(args.n_cpus, os.cpu_count() or 1)

    main(n_cpus)
