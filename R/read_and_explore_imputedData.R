# Read data
pfad_o <- "/home/joern/Aktuell/PaclitaxelPainLipidomics/"
pfad_u1 <- "01Transformierte/sissignano/paclitaxel/results/"

###################################### UCT #######################################################
paclitaxel_uct_imputed <- read.csv(paste0(pfad_o, pfad_u1, "paclitaxel_uct_imputed.csv"))
rownames(paclitaxel_uct_imputed) <- paste0(
  paclitaxel_uct_imputed$Patientennummer, "_Day_", paclitaxel_uct_imputed$Probe.1.oder.2,
  "_NP_", ifelse(paclitaxel_uct_imputed$Probe.1.oder.2 > 1, paclitaxel_uct_imputed$Neuropathie..0.5.., "Baseline")
)
names(paclitaxel_uct_imputed)

nonLipidVariableNames <- c(
  "Messung", "Patientennummer", "Initialen", "Geburtsjahr", "Probe.1.oder.2",
  "Zyklus", "Neuropathie..0.5..", "Datum.der.Blutentnahme",
  "Baehandlungsregime...Dosieung.Paclitaxel..mg.m.2."
)
LipidVariableNames <- setdiff(names(paclitaxel_uct_imputed), nonLipidVariableNames)

table(paclitaxel_uct_imputed$Neuropathie..0.5..)
paclitaxel_uct_imputed$Neuropathie <- ifelse(paclitaxel_uct_imputed$Neuropathie..0.5.. == 0, 0, 1)
table(paclitaxel_uct_imputed$Neuropathie)

# # PC-corr analysis
# source(paste0("/home/joern/Dokumente/BiomarkerLipide/05OurPublication/RevisionSciReportsV2/BilderDooferProgramme/pccorrv2.R"))
# library(xlsx)
# # notwendige parameter fuer pc-corr
# PCcorrDaten <- subset(paclitaxel_uct_imputed, select = c("Messung", LipidVariableNames))
# dim(PCcorrDaten)
# head(PCcorrDaten)
# table(PCcorrDaten$Messung)
#
# sample_labels = PCcorrDaten$Messung
# feat_names = names(PCcorrDaten[2:ncol(PCcorrDaten)])
# sample_names = row.names(PCcorrDaten)
#
# Vdiff <- PC_corr_v2(as.matrix(PCcorrDaten[names(PCcorrDaten) %in% feat_names]),
#                     sample_labels, feat_names, sample_names, "no")
#
#
# PCcorrDaten <- subset(paclitaxel_uct_imputed, select = c("Neuropathie", LipidVariableNames))
# dim(PCcorrDaten)
# head(PCcorrDaten)
# table(paclitaxel_uct_imputed$Neuropathie)
#
# sample_labels = PCcorrDaten$Neuropathie
# feat_names = names(PCcorrDaten[2:ncol(PCcorrDaten)])
# sample_names = row.names(PCcorrDaten)
#
# Vdiff <- PC_corr_v2(as.matrix(PCcorrDaten[names(PCcorrDaten) %in% feat_names]),
#                     sample_labels, feat_names, sample_names, "no")

# Data transformation

paclitaxel_uct_imputed_log <- paclitaxel_uct_imputed
paclitaxel_uct_imputed_log[names(paclitaxel_uct_imputed_log) %in% LipidVariableNames] <- lapply(paclitaxel_uct_imputed_log[names(paclitaxel_uct_imputed_log) %in% LipidVariableNames], log10)

# Plot data
library(ComplexHeatmap)
library(cluster)
dfLipids <- paclitaxel_uct_imputed_log[names(paclitaxel_uct_imputed_log) %in% LipidVariableNames]
dfLipids_scaled <- scale(dfLipids)
Heatmap(as.matrix(dfLipids_scaled),
  cluster_rows = F,
  clustering_method_columns = "ward.D2",
  col = rev(topo.colors(123)),
  row_dend_width = unit(4, "cm"),
  column_dend_height = unit(4, "cm"),
  show_heatmap_legend = T,
  row_names_gp = gpar(fontsize = 8),
  column_names_gp = gpar(fontsize = 5)
)

library(ggplot2)
library(ggthemes)
library(ggforce)

# Both days lumped together
dim(dfLipids)
dfLipids_long <- reshape2::melt(dfLipids)
dfLipids_long$figpanel <- 2
nPanels <- 2
nPerPanel <- floor(dim(dfLipids2)[2] / nPanels)
dfLipids2_long$figpanel <- 2
SplitsFig <- split(LipidVariableNames, ceiling(seq_along(LipidVariableNames)/nPerPanel))
for (i in 1:length(SplitsFig)) {
  dfLipids_long$figpanel[which(dfLipids_long$variable %in% SplitsFig[[i]])] <- i
}
# dfLipids_long$figpanel <- 2
# dfLipids_long$figpanel[1:(dim(dfLipids)[1] * floor(dim(dfLipids)[2] / 2))] <- 1

ggplot(data = dfLipids_long, aes(x = variable, y = value)) +
  geom_violin(color = "lightsalmon4") +
  #  geom_boxplot(outlier.shape = NA, color = "dodgerblue4") +
  geom_sina(size = .01, color = "lightsalmon2",alpha=0.4) +
  facet_wrap(~figpanel, scale = "free_x", ncol = 1) +
  theme_grey() +
  theme(
    strip.background = element_blank(), strip.text.x = element_blank(),
    axis.text.x = element_text(size = 5, angle = 90, vjust = 0.5, hjust = 1, color = "black")
  ) +
  labs(x = NULL, y = "log concentration")

# Days separated
dfLipids2 <- dfLipids
dim(dfLipids2)
dfLipids2$Samplingday <- 2
dfLipids2$Samplingday[grep("Base", rownames(dfLipids2))] <- 1
dfLipids2_long <- reshape2::melt(dfLipids2, id.vars = "Samplingday")
nPanels <- 4
nPerPanel <- floor(dim(dfLipids2)[2] / nPanels)
dfLipids2_long$figpanel <- 2
SplitsFig <- split(LipidVariableNames, ceiling(seq_along(LipidVariableNames)/nPerPanel))
for (i in 1:length(SplitsFig)) {
  dfLipids2_long$figpanel[which(dfLipids2_long$variable %in% SplitsFig[[i]])] <- i
}
table(dfLipids2_long$figpanel)
ggplot(data = dfLipids2_long, aes(x = variable, y = value, color = factor(Samplingday))) +
  #geom_violin(lwd=.2) +
    geom_boxplot(outlier.shape = NA, lwd = .2) +
  geom_sina(alpha=0.2, size = .1) +
  facet_wrap(~figpanel, scale = "free_x", ncol = 1) +
  theme_grey() +
  scale_color_colorblind() +
  scale_fill_colorblind() +
  theme( legend.position = c(.1,.58), legend.direction = "horizontal",
         legend.background = element_rect(colour = "transparent", fill = alpha("white", 0.6)),
    strip.background = element_blank(), strip.text.x = element_blank(),
    axis.text.x = element_text(size = 5, angle = 90, vjust = 0.5, hjust = 1, color = "black")
  ) +
  labs(x = NULL, y = "log concentration", color = "Sampling day")



# Einige Tests von Lipiden, die sich laut vorwissen regulieren koennten
# wilcox.test(paclitaxel_uct_imputed_log$EpOME_910 ~ paclitaxel_uct_imputed_log$Probe.1.oder.2)
# wilcox.test(paclitaxel_uct_imputed_log$EpOME_910[paclitaxel_uct_imputed_log$Probe.1.oder.2 == 2] ~
#               paclitaxel_uct_imputed_log$Neuropathie[paclitaxel_uct_imputed_log$Probe.1.oder.2 == 2])
# wilcox.test(paclitaxel_uct_imputed_log$HETE_5 ~ paclitaxel_uct_imputed_log$Probe.1.oder.2)
# wilcox.test(paclitaxel_uct_imputed_log$HETE_5[paclitaxel_uct_imputed_log$Probe.1.oder.2 == 2] ~
#               paclitaxel_uct_imputed_log$Neuropathie[paclitaxel_uct_imputed_log$Probe.1.oder.2 == 2])
#

# write.csv(dfLipids_scaled, "dfLipids_scaled.csv")



###################################### SANA #######################################################
paclitaxel_sana_imputed <- read.csv(paste0(pfad_o, pfad_u1, "paclitaxel_sana_imputed.csv"))
rownames(paclitaxel_sana_imputed) <- paste0(
  paclitaxel_sana_imputed$Patientennummer, "_Day_", paclitaxel_sana_imputed$Probe.1.oder.2,
  "_NP_", ifelse(paclitaxel_sana_imputed$Probe.1.oder.2 > 1, paclitaxel_sana_imputed$Neuropathie..0.5.., "Baseline")
)
names(paclitaxel_sana_imputed)

table(paclitaxel_sana_imputed$Neuropathie..0.5..)
paclitaxel_sana_imputed$Neuropathie <- ifelse(paclitaxel_sana_imputed$Neuropathie..0.5.. == 0, 0, 1)
table(paclitaxel_sana_imputed$Neuropathie)

paclitaxel_sana_imputed_log <- paclitaxel_sana_imputed
paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames] <- lapply(paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames], log10)
dfLipids_sana <- paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames]
names(dfLipids_sana)
dfLipids_sana_scaled <- data.frame(scale(dfLipids_sana))
names(dfLipids_sana_scaled)
