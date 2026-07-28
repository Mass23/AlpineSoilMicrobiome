## 🌍 Global Alpine Microbiome Study

### 📊 Data

Data are sourced from the **MicrobeAtlas platform** ([https://microbeatlas.org](https://microbeatlas.org)), which aggregates **microbial community profiles with global metadata** ([ScienceDirect][1])

Based global soil microbiome datasets generated for:

* Reproducible Propagation of Species-Rich Soil Bacterial Communities / Habitat filtering more than microbiota origin controls microbiome transplant outcomes in soil (https://zenodo.org/records/15236630)

Along with other global datasets:
- SoilGrids v2.0 for soil properties
- CHELSA v2.1 (chelsa-climate.org) for bioclimatic variables
- [] for topographic variables
- NDVI from Moeslund et al. (2022) Ecography doi:10.1111/ecog.05012

---

### 🚀 Step 0b — Environmental covariate extraction


#### Installation



#### Usage



#### Data layers added

| Prefix | Source | Resolution | Notes |
|--------|--------|-----------|-------|
| `chelsa_bio1` … `chelsa_bio19` | CHELSA v2.1 | ~1 km | BioClim 1981–2010; read remotely via VSICURL |
| `srtm_elev`, `srtm_slope`, `srtm_aspect`, `srtm_focal_*` | CGIAR-CSI SRTM v4.1 | ~90 m | focal mean/std at 10 m, 100 m, 1 000 m scales |
| `aster_elev`, `aster_slope`, `aster_aspect`, `aster_focal_*` | ASTER GDEM v3 (OpenTopography) | ~30 m | requires `OPENTOPOGRAPHY_API_KEY` |
| `ndvi` | Local raster (Ecography doi:10.1111/ecog.05012) | varies | place GeoTIFF in `data/ndvi/` |
| `worldcover_class`, `worldcover_name` | ESA WorldCover 2021 v200 | 10 m | read remotely via VSICURL |
| `soilgrids_phh2o_*`, `soilgrids_bdod_*`, … | SoilGrids v2.0 REST API | ~250 m | pH, bulk density, SOC, clay, sand |

#### Optional: ASTER GDEM v3 and SRTM 30 m

Register for a **free** API key at <https://portal.opentopography.org> and
export it before running:

```bash
export OPENTOPOGRAPHY_API_KEY=your_key_here
python3 0b_get_climatic_topo_land_data.py
```

Without the key, SRTM 30 m and ASTER GDEM are skipped; CGIAR-CSI SRTM 90 m
is used instead.

#### NDVI raster

Download the global NDVI raster from:
> Moeslund J.E. et al. (2022) *Ecography* <https://doi.org/10.1111/ecog.05012>

Place the GeoTIFF in `data/ndvi/` (any `*.tif` or `*.tiff` filename is
accepted).

#### Reproducibility log

Every run appends dataset names, versions, source URLs, access timestamps,
SHA-256 hashes of downloaded files, and library versions to
`data/api_data_log.txt`.

---

### 🧪 Methods (overview)

* Define **alpine vs non-alpine** samples (metadata: elevation / habitat)

* Filter & normalize abundance data (relative abundance / CLR)

* **Alpha diversity**: Shannon, Simpson, richness

* **Beta diversity**: Bray–Curtis / Aitchison + ordination, PERMANOVA

* **Differential abundance**: ANCOM-BC / DESeq2 / LinDA

* **Species distribution models**:

  * GLM / Random Forest / MaxEnt
  * Predictors: temperature, elevation, climate

[1]: https://www.sciencedirect.com/science/article/pii/S009286742600108X?utm_source=chatgpt.com "The MicrobeAtlas database: Global trends and insights into Earth’s microbial ecosystems - ScienceDirect"
