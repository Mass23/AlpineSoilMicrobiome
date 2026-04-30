library(dplyr)
library(ggplot2)
library(purrr)
library(maps)
library(tidyr)
library(raster)
library(terra)
library(elevatr)
library(data.table)
library(sf)

setwd('~/Documents/MACE/AlpineSoilMicrobiome')

sample_env = read.table('data/microbeatlas/microbe_atlas_meta_data.tsv', sep='\t', header = T)
sample_env = sample_env %>%
  separate(SampleID, into = c("Reads_acc", "Sample_acc"), sep = "\\.")

sample_latlon = read.csv('data/microbeatlas/samples.info.latlon.parsed', sep= '\t', header = T, quote = "")
sample_env_filtered <- sample_env %>%
  left_join(
    sample_latlon %>% 
      dplyr::select(X.sampleId, parsedLat, parsedLon) %>%
      rename(Sample_acc = X.sampleId),
    by = "Sample_acc"
  ) %>%
  rename(
    Latitude  = parsedLat,
    Longitude = parsedLon
  )

sample_env_filtered = sample_env_filtered %>% filter(Latitude != 'None', Longitude != 'None')
sample_env_filtered$Latitude = as.numeric(sample_env_filtered$Latitude)
sample_env_filtered$Longitude = as.numeric(sample_env_filtered$Longitude)

sample_env_filtered = sample_env_filtered %>% filter(Longitude > -180)
sample_env_filtered = sample_env_filtered %>% filter(Longitude < 180)
sample_env_filtered = sample_env_filtered %>% filter(Latitude > -90)
sample_env_filtered = sample_env_filtered %>% filter(Latitude < 90)

# Categorise Alpine samples
# Based on this: https://figshare.com/articles/dataset/Global_distribution_and_bioclimatic_characterization_of_alpine_biomes/11710002?file=33157427
v <- vect("data/global_alpine_30m_v1_1/global_alpine_30m_v1_1.shp")
pts <- vect(sample_env_filtered %>% dplyr::select("Longitude", "Latitude") %>% rename(lon=Longitude, lat=Latitude), crs = "EPSG:4326")
alpine <- extract(v, pts)
sample_env_filtered$Alpine <- ifelse(is.na(alpine[,4]), 0L, 1L)
sample_env_filtered = sample_env_filtered[!is.na(sample_env_filtered$Alpine),]

# Keep only samples that fall on Land
# Based on this: https://www.naturalearthdata.com/downloads/10m-physical-vectors/
land <- vect("data/ne_10m_land/ne_10m_land.shp")
pts <- vect(sample_env_filtered, geom = c("Longitude", "Latitude"), crs = "EPSG:4326")
on_land <- extract(land, pts)
to_keep = na.omit(on_land[on_land[,2] == 'Land',1])
sample_env_filtered <- sample_env_filtered[on_land[,2] == 'Land',]

# How many alpine/non-alpine samples?
sample_env_filtered$Alpine = as.factor(sample_env_filtered$Alpine)  
sample_env_filtered = na.omit(sample_env_filtered)
table(sample_env_filtered$Alpine)

shannon = read.table('data/microbeatlas/microbe_atlas_shannon_diversity.tsv', sep='\t', header = T)
shannon <- shannon %>%
  mutate(Sample_acc = sub("^[^.]+\\.", "", SampleID)) %>% filter(Sample_acc %in% sample_env_filtered$Sample_acc)
mean(sample_env_filtered$Sample_acc == shannon$Sample_acc)

sample_env_filtered_shannon = sample_env_filtered
sample_env_filtered_shannon$Shannon = shannon$Shannon.diversity
   
write.csv(sample_env_filtered_shannon, file = 'data/sample_data_filtered.csv', quote = F, row.names = F)
