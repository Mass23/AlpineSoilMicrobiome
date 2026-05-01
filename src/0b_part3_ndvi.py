#!/usr/bin/env python3
"""NDVI (Landsat 8) per-point extraction.

Method (per user spec):
- Compute maximum NDVI value at each 30x30m pixel using Landsat 8 Annual Greenest-Pixel images (2013-2018) in Google Earth Engine.
- Create composite with per-pixel max NDVI.
- Upscale final composite to 30 arc-sec resolution.
- Remove artifacts: NDVI < 0 or NDVI > 1.

This script:
- samples a local exported GeoTIFF under data/ndvi/
- if missing, logs and prints an exact GEE snippet to generate it.

Output:
  data/ndvi/ndvi_points.csv

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import DATA_DIR, log_api_data, load_points, sample_raster_at_points, sha256_file


NDVI_TIF = DATA_DIR / "ndvi" / "ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif"


GEE_SNIPPET = r"""
// Landsat 8 max NDVI (greenest pixel) composite, 2013-2018, upscaled to 30 arc-sec.
// Paste into https://code.earthengine.google.com/

var roi = ee.Geometry.Rectangle([-180, -60, 180, 85], null, false);

function maskL8sr(image) {
  // Landsat 8 SR QA mask (C01 SR). This is a simple mask; adjust if needed.
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

var ndviMax = col.max();

// Remove artifacts: clamp to [0,1]
ndviMax = ndviMax.where(ndviMax.lt(0), 0);
ndviMax = ndviMax.where(ndviMax.gt(1), 1);

// Upscale to 30 arc-sec (~0.008333333 degrees)
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

// After export completes, download and place at:
// data/ndvi/ndvi_greenestpixel_landsat8_2013_2018_30arcsec.tif
"""


def run(input_csv: Path = DATA_DIR / "sample_data_filtered.csv") -> pd.DataFrame:
    ndvi_dir = DATA_DIR / "ndvi"
    ndvi_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(input_csv)
    valid = df["__valid_latlon"].to_numpy()
    lats = df["Latitude"].to_numpy(float)
    lons = df["Longitude"].to_numpy(float)

    out = pd.DataFrame(index=df.index)

    if not NDVI_TIF.exists():
        log_api_data(
            [
                "NDVI: missing local GeoTIFF export.",
                f"Expected: {NDVI_TIF}",
                "Generate it using Google Earth Engine with the following snippet:",
                GEE_SNIPPET,
            ],
            script_name=Path(__file__).name,
        )
        print("NDVI GeoTIFF not found. Please export from Google Earth Engine and place at:")
        print(f"  {NDVI_TIF}")
        print("GEE snippet (also logged to data/api_data_log.txt):")
        print(GEE_SNIPPET)
        raise SystemExit(2)

    log_api_data(
        [
            "NDVI: sampling local raster",
            f"Path: {NDVI_TIF}",
            f"sha256: {sha256_file(NDVI_TIF)}",
            "Method: Landsat 8 SR 2013-2018 max NDVI, upscaled to 30 arc-sec, artifacts removed.",
        ],
        script_name=Path(__file__).name,
    )

    vals = np.full(len(df), np.nan)
    vals[valid] = sample_raster_at_points(NDVI_TIF, lons[valid], lats[valid])
    # enforce artifact removal
    vals[(vals < 0) | (vals > 1)] = np.nan
    out["ndvi_greenestpixel_l8_2013_2018_30arcsec"] = vals

    out_path = ndvi_dir / "ndvi_points.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run()
