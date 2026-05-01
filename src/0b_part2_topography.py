#!/usr/bin/env python3
"""Topography extraction from user-provided DEM.

User requirement:
- Use a high-resolution DEM; user will download locally.
- Compute elevation, slope, aspect.
- Compute focal slope/aspect at 100m and 1000m.

This script expects a DEM GeoTIFF at:
  data/topography/dem_aw3d30.tif   (recommended naming)

Outputs:
  data/topography/topography_points.csv

Temporary products:
- Intermediate rasters may be written under data/topography/tmp_*/ and will be
  deleted after point extraction.

"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_DIR, log_api_data, load_points, require_rasterio, sample_raster_at_points, sha256_file


DEM_DEFAULT = DATA_DIR / "topography" / "dem_aw3d30.tif"


def _compute_slope_aspect(dem: np.ndarray, res_x: float, res_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute slope (degrees) and aspect (degrees) using simple Horn gradient."""
    # Horn's method kernel derivatives
    dzdx = (
        (dem[0:-2, 2:] + 2 * dem[1:-1, 2:] + dem[2:, 2:])
        - (dem[0:-2, 0:-2] + 2 * dem[1:-1, 0:-2] + dem[2:, 0:-2])
    ) / (8 * res_x)
    dzdy = (
        (dem[2:, 0:-2] + 2 * dem[2:, 1:-1] + dem[2:, 2:])
        - (dem[0:-2, 0:-2] + 2 * dem[0:-2, 1:-1] + dem[0:-2, 2:])
    ) / (8 * res_y)

    slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    aspect = np.degrees(np.arctan2(dzdy, -dzdx))
    aspect = np.where(aspect < 0, 90.0 - aspect, 360.0 - aspect + 90.0)
    aspect = np.mod(aspect, 360.0)

    return slope, aspect


def run(
    input_csv: Path = DATA_DIR / "sample_data_filtered.csv",
    dem_path: Path = DEM_DEFAULT,
) -> pd.DataFrame:
    require_rasterio()
    import rasterio

    topo_dir = DATA_DIR / "topography"
    topo_dir.mkdir(parents=True, exist_ok=True)

    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM not found: {dem_path}\n"
            "Download AW3D30 (or another DEM) and place a merged GeoTIFF at that path."
        )

    df = load_points(input_csv)
    valid = df["__valid_latlon"].to_numpy()
    lats = df["Latitude"].to_numpy(float)
    lons = df["Longitude"].to_numpy(float)

    log_api_data(
        [
            "Topography: user-provided DEM workflow",
            f"DEM path: {dem_path}",
            f"DEM sha256: {sha256_file(dem_path)}",
            "Derivatives: elevation, slope, aspect",
            "Focals: slope/aspect focal means at 100m and 1000m (approximated by pixel windows)",
        ],
        script_name=Path(__file__).name,
    )

    # We compute slope/aspect and focal metrics in a temp dir, then sample at points.
    tmpdir = Path(tempfile.mkdtemp(prefix="topo_tmp_", dir=str(topo_dir)))

    try:
        with rasterio.open(dem_path) as src:
            dem = src.read(1).astype("float64")
            nodata = src.nodata
            if nodata is not None:
                dem = np.where(dem == nodata, np.nan, dem)

            # resolution in projected units (if geographic, this is degrees; for best results provide projected DEM)
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)

            slope, aspect = _compute_slope_aspect(dem, res_x=res_x, res_y=res_y)

            # Save slope/aspect rasters
            slope_path = tmpdir / "slope_deg.tif"
            aspect_path = tmpdir / "aspect_deg.tif"

            meta = src.meta.copy()
            meta.update(dtype="float32", nodata=np.nan)

            with rasterio.open(slope_path, "w", **meta) as dst:
                dst.write(slope.astype("float32"), 1)
            with rasterio.open(aspect_path, "w", **meta) as dst:
                dst.write(aspect.astype("float32"), 1)

        # Focal: approximate 100m / 1000m windows
        # If DEM is in meters, window_radius_px = round(radius_m / pixel_size_m)
        # For geographic DEMs in degrees, user should provide projected DEM for correct meters.
        def focal_mean(in_path: Path, radius: float, out_path: Path):
            from scipy.ndimage import uniform_filter
            with rasterio.open(in_path) as src:
                arr = src.read(1).astype("float64")
                px = abs(src.transform.a)
                # Treat `radius` in same units as px
                r_px = max(1, int(round(radius / px)))
                size = 2 * r_px + 1
                out = uniform_filter(arr, size=size, mode="nearest")
                meta = src.meta.copy()
                meta.update(dtype="float32")
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(out.astype("float32"), 1)

        slope_100 = tmpdir / "slope_100m_mean.tif"
        slope_1000 = tmpdir / "slope_1000m_mean.tif"
        aspect_100 = tmpdir / "aspect_100m_mean.tif"
        aspect_1000 = tmpdir / "aspect_1000m_mean.tif"

        focal_mean(slope_path, 100.0, slope_100)
        focal_mean(slope_path, 1000.0, slope_1000)
        focal_mean(aspect_path, 100.0, aspect_100)
        focal_mean(aspect_path, 1000.0, aspect_1000)

        # Sample point values
        out = pd.DataFrame(index=df.index)
        elev = np.full(len(df), np.nan)
        elev[valid] = sample_raster_at_points(dem_path, lons[valid], lats[valid])
        out["dem_elevation"] = elev

        for name, path in [
            ("dem_slope_deg", slope_path),
            ("dem_aspect_deg", aspect_path),
            ("dem_slope_deg_mean_100m", slope_100),
            ("dem_slope_deg_mean_1000m", slope_1000),
            ("dem_aspect_deg_mean_100m", aspect_100),
            ("dem_aspect_deg_mean_1000m", aspect_1000),
        ]:
            vals = np.full(len(df), np.nan)
            vals[valid] = sample_raster_at_points(path, lons[valid], lats[valid])
            out[name] = vals

        out_path = topo_dir / "topography_points.csv"
        out.to_csv(out_path, index=False)
        return out

    finally:
        # Delete temp tiffs
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    run()
