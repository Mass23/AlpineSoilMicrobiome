library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggpubr)
library(performance)
library(mgcv)

setwd('~/Documents/MACE/AlpineSoilMicrobiome')
meta_all = read.csv('data/sample_data_filtered.csv')
meta_all$Alpine[meta_all$Alpine == 1] = 'Alpine'
meta_all$Alpine[meta_all$Alpine == 0] = 'Lowland'

meta_all$Alpine_regions = ''
meta_all$Alpine_regions[(meta_all$Longitude < 0) & (meta_all$Latitude < 0)] = 'South America'
meta_all$Alpine_regions[(meta_all$Longitude > 0) & (meta_all$Latitude < 0)] = 'Oceania'
meta_all$Alpine_regions[(meta_all$Longitude < -20) & (meta_all$Latitude > 0)] = 'North and Central America'
meta_all$Alpine_regions[(meta_all$Longitude > -20) & (meta_all$Longitude < 30) & (meta_all$Latitude > 0)] = 'Europe'
meta_all$Alpine_regions[(meta_all$Longitude > 30) & (meta_all$Latitude > 0)] = 'Asia'
meta_all$Alpine_regions[meta_all$Alpine == 'Lowland'] = 'Lowland'
table(meta_all$Alpine_regions, meta_all$Alpine)

meta_all$Alpine_regions <- factor(meta_all$Alpine_regions, levels = c("Lowland", "Oceania", "Europe", "Asia", "North and Central America", "South America"))
meta_all$Alpine <- factor(meta_all$Alpine, levels = c("Lowland", "Alpine"))

meta_all %>% filter(Alpine == 'Alpine') %>% group_by(Alpine_regions) %>% 
  summarise(mean = mean(Shannon),
            median = median(Shannon),
            iqr = quantile(Shannon, probs = 0.75) - quantile(Shannon, probs = 0.25))

ggplot(meta_all, aes(x=Alpine, y=Shannon, colour=Alpine)) + geom_boxplot() + stat_compare_means()
ggplot(meta_all, aes(x=Alpine_regions, y=Shannon, fill=Alpine)) + geom_boxplot() + stat_compare_means(label.y = 14) + stat_compare_means(comparisons = list(c('Lowland', 'Oceania'),
                                                                                                                                                  c('Lowland', 'Europe'),
                                                                                                                                                  c('Lowland', 'Asia'),
                                                                                                                                                  c('Lowland', 'North and Central America'),
                                                                                                                                                  c('Lowland', 'South America')), label = 'p.signif') +
  theme_bw() + xlab('') + theme(legend.position = 'none') + scale_fill_manual(values = c('#8F8287','#0974E0'))

ggplot(meta_all, aes(x=Latitude, y=Shannon, colour=Alpine)) + geom_point(alpha=0.01) + geom_smooth(formula = y ~ s(x, k=3))



mod_shannon = gam(data = meta_all, formula = Shannon ~ s(Latitude, Longitude, bs='sos') + Alpine)

plot(mod_shannon)
summary(mod_shannon)

WorldData <- ggplot2::map_data('world') %>% fortify

ggplot() +
  geom_map(data = WorldData, map = WorldData,
           aes(long, lat, map_id = region),
           fill = "#DBDBD8", colour = "#DBDBD8", size=0.2) +
  coord_map("rectangular", lat0=0, xlim=c(-180,180), ylim=c(-90, 90)) + xlab('') + ylab('') +
  geom_point(data=meta_all %>% arrange(Alpine), aes(x=Longitude, y=Latitude, colour=Alpine, alpha=Alpine)) + 
  scale_alpha_discrete(c(0.01,1), guide = 'none') + 
  scale_colour_manual(values = c('#8F8287','#0974E0')) + 
  theme_bw() + theme(legend.title = element_text(size=9), 
                     legend.text=element_text(size=6), 
                     axis.title=element_text(size=8), 
                     legend.position="bottom",
                     axis.title.x=element_blank(),
                     axis.text.x=element_blank(),
                     axis.ticks.x=element_blank(),
                     axis.title.y=element_blank(),
                     axis.text.y=element_blank(),
                     axis.ticks.y=element_blank(),
                     legend.margin=margin(t = 0, unit='cm')) + 
  guides(colour = guide_legend(override.aes = list(alpha=1, size=3)), shape = guide_legend(override.aes = list(linewdith=3)))

