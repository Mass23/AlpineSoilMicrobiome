## 🌍 Global Alpine Microbiome Study

### 📊 Data

Data are sourced from the **MicrobeAtlas platform** ([https://microbeatlas.org](https://microbeatlas.org)), which aggregates **microbial community profiles with global metadata** ([ScienceDirect][1])

Based global soil microbiome datasets generated for:

* Reproducible Propagation of Species-Rich Soil Bacterial Communities / Habitat filtering more than microbiota origin controls microbiome transplant outcomes in soil (https://zenodo.org/records/15236630)

Along with other global datasets:
- CHELSA (chelsa.org) for bioclimatic variables
- Google Earth Engine for topgraphical variables
- ... for soil and vegetation variables

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
