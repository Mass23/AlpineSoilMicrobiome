#!/usr/bin/env python3
"""Soil properties from SoilGrids v2.

Currently implemented:
- pH in water (phh2o), depth 0-5cm, mean value

Uses multiprocessing (12 CPUs by default) for point queries.

Output:
  data/soil_properties/soil_points.csv

"""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .utils import DATA_DIR, clamp_cpus, log_api_data, load_points


DEFAULT_CPUS = 12


def soilgrids_point_query(lat: float, lon: float, properties: list[str]) -> dict:
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


def extract_phh2o_0_5cm_mean(resp: dict):
    try:
        for layer in resp["properties"]["layers"]:
            if layer.get("name") == "phh2o":
                for d in layer["depths"]:
                    if d.get("label") == "0-5cm":
                        return float(d["values"]["mean"])
    except Exception:
        return np.nan
    return np.nan


def _worker(lat: float, lon: float) -> float:
    try:
        resp = soilgrids_point_query(lat, lon, properties=["phh2o"])
        return extract_phh2o_0_5cm_mean(resp)
    except Exception:
        return np.nan


def run(input_csv: Path = DATA_DIR / "sample_data_filtered.csv", cpus: int = DEFAULT_CPUS) -> pd.DataFrame:
    soil_dir = DATA_DIR / "soil_properties"
    soil_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(input_csv)
    valid = df["__valid_latlon"].to_numpy()

    cpus_use = clamp_cpus(cpus)

    log_api_data(
        [
            "Soil: SoilGrids v2.0 point query",
            "Endpoint: https://rest.isric.org/soilgrids/v2.0/properties/query",
            "Property: phh2o, depth 0-5cm, value mean",
            f"CPUs requested: {cpus}; using: {cpus_use}",
            "Note: SoilGrids pH values may be scaled (often pH*10). This script stores raw response.",
        ],
        script_name=Path(__file__).name,
    )

    out = pd.DataFrame(index=df.index)
    vals = np.full(len(df), np.nan)

    idxs = np.where(valid)[0]
    coords = [(float(df.at[i, "Latitude"]), float(df.at[i, "Longitude"])) for i in idxs]

    with mp.get_context("spawn").Pool(processes=cpus_use) as pool:
        for i, v in tqdm(zip(idxs, pool.starmap(_worker, coords)), total=len(idxs), desc="SoilGrids phh2o"):
            vals[i] = v

    out["soilgrids_phh2o_0_5cm_mean_raw"] = vals

    out_path = soil_dir / "soil_points.csv"
    out.to_csv(out_path, index=False)
    return out


if __name__ == "__main__":
    run()
