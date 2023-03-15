###################################### paths #######################################################
pfad_o <- "/home/joern/Aktuell/PaclitaxelPainLipidomics/"
pfad_u1 <- "01Transformierte/sissignano/paclitaxel/results/"

###################################### libraries #######################################################
library(ComplexHeatmap)
library(cluster)

library(ggplot2)
library(ggthemes)
library(ggforce)
library(ggpubr)
library(gridGraphics)
library(grid)
library(gridExtra)
library(cowplot)

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
length(LipidVariableNames)

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

dfLipids <- paclitaxel_uct_imputed_log[names(paclitaxel_uct_imputed_log) %in% LipidVariableNames]
dfLipids_scaled <- scale(dfLipids)

HM_uct <- 
  grid.grabExpr(draw(
Heatmap(as.matrix(dfLipids_scaled),
        cluster_rows = F,
        cluster_columns =  F,
        clustering_method_columns = "ward.D2",
  col = rev(topo.colors(123)),
  row_dend_width = unit(4, "cm"),
  column_dend_height = unit(4, "cm"),
  show_heatmap_legend = F,
  row_names_gp = gpar(fontsize = 6),
  column_names_gp = gpar(fontsize = 3),
  column_title = "Lipid markers, cohort 1"
)
))

# Both days lumped together
dim(dfLipids)
dfLipids_long <- reshape2::melt(dfLipids)
dfLipids_long$figpanel <- 2
nPanels <- 2
nPerPanel <- ceiling(dim(dfLipids)[2] / nPanels)
dfLipids_long$figpanel <- 2
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
nPerPanel <- ceiling(dim(dfLipids2)[2] / nPanels)
dfLipids2_long$figpanel <- 2
SplitsFig <- split(LipidVariableNames, ceiling(seq_along(LipidVariableNames)/nPerPanel))
for (i in 1:length(SplitsFig)) {
  dfLipids2_long$figpanel[which(dfLipids2_long$variable %in% SplitsFig[[i]])] <- i
}
table(dfLipids2_long$figpanel)
ggplot(data = dfLipids2_long, aes(x = variable, y = value, color = factor(Samplingday))) +
  geom_violin(lwd=.2, width = 2, position = position_dodge(1)) +
  #  geom_boxplot(outlier.shape = NA, lwd = .2, position = position_dodge(1)) +
  geom_sina(alpha=0.2, size = .1, position = position_dodge(1)) +
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
# Correction for one wrong patient number 
paclitaxel_sana_imputed$Patientennummer[paclitaxel_sana_imputed$Initialen == "K-W"] <- 166

length(unique(paclitaxel_sana_imputed$Patientennummer))
table(paclitaxel_sana_imputed$Probe.1.oder.2)

rownames(paclitaxel_sana_imputed) <- paste0(
  paclitaxel_sana_imputed$Patientennummer, "_Day_", paclitaxel_sana_imputed$Probe.1.oder.2,
  "_NP_", ifelse(paclitaxel_sana_imputed$Probe.1.oder.2 > 1, paclitaxel_sana_imputed$Neuropathie..0.5.., "Baseline")
)
names(paclitaxel_sana_imputed)

table(paclitaxel_sana_imputed$Neuropathie..0.5..)
paclitaxel_sana_imputed$Neuropathie <- ifelse(paclitaxel_sana_imputed$Neuropathie..0.5.. == 0, 0, 1)
table(paclitaxel_sana_imputed$Neuropathie)

paclitaxel_sana_imputed_log <- paclitaxel_sana_imputed
paclitaxel_sana_imputed_log <- paclitaxel_sana_imputed_log[paclitaxel_sana_imputed_log$Patientennummer %in% 
                                                             paclitaxel_sana_imputed_log$Patientennummer[duplicated(paclitaxel_sana_imputed_log$Patientennummer)], ]


paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames] <- lapply(paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames], log10)
dfLipids_sana <- paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames]
names(dfLipids_sana)
dfLipids_sana_scaled <- data.frame(scale(dfLipids_sana))
names(dfLipids_sana_scaled)

LipidVariableNames_sana <- intersect(names(dfLipids_sana_scaled), LipidVariableNames)
length(LipidVariableNames)
length(LipidVariableNames_sana)

setdiff( LipidVariableNames,LipidVariableNames_sana)


# Plot data
dfLipids_sana <- paclitaxel_sana_imputed_log[names(paclitaxel_sana_imputed_log) %in% LipidVariableNames_sana]
col_fun = circlize::colorRamp2(c(-2, 0, 2), c("dodgerblue", "white", "chartreuse3"))

HM_sana <- 
  grid.grabExpr(draw(
        Heatmap(as.matrix(dfLipids_sana_scaled),
        cluster_rows = F,
        cluster_columns =  F,
        clustering_method_columns = "ward.D2",
        col = rev(topo.colors(123)),
        row_dend_width = unit(4, "cm"),
        column_dend_height = unit(4, "cm"),
        show_heatmap_legend = T,
        row_names_gp = gpar(fontsize = 6),
        column_names_gp = gpar(fontsize = 3),
        column_title = "Lipid markers, cohort 2",
        heatmap_legend_param = list(
          title = "Standardized concentration", 
          #at = c(-2, 0, 2), labels = c("neg_two", "zero", "pos_two"), 
          legend_direction = "vertical", 
          legend_width = unit(5, "cm")  ,
          title_position = "leftcenter-rot")
)
))


Abberantprobes <- 
  rownames(dfLipids_sana_scaled)[53:nrow(dfLipids_sana_scaled)]

paclitaxel_sana_imputed_log <- paclitaxel_sana_imputed_log[1:52,]
dfLipids_sana_scaled_new <- data.frame(scale(dfLipids_sana[1:52,]))

length(unique(paclitaxel_sana_imputed_log$Patientennummer))
table(paclitaxel_sana_imputed_log$Neuropathie..0.5..)

# Plot data
Heatmap(as.matrix(dfLipids_sana_scaled_new),
        cluster_rows = F,
        clustering_method_columns = "ward.D2",
        col = rev(topo.colors(123)),
        row_dend_width = unit(4, "cm"),
        column_dend_height = unit(4, "cm"),
        show_heatmap_legend = T,
        row_names_gp = gpar(fontsize = 8),
        column_names_gp = gpar(fontsize = 5)
)

# Both days lumped together
dim(dfLipids_sana)
dfLipids_sana_long <- reshape2::melt(dfLipids_sana)
dfLipids_sana_long$figpanel <- 2
nPanels <- 2
nPerPanel <- ceiling(dim(dfLipids_sana)[2] / nPanels)
dfLipids_sana_long$figpanel <- 2
SplitsFig <- split(LipidVariableNames_sana, ceiling(seq_along(LipidVariableNames_sana)/nPerPanel))
for (i in 1:length(SplitsFig)) {
  dfLipids_sana_long$figpanel[which(dfLipids_sana_long$variable %in% SplitsFig[[i]])] <- i
}
# dfLipids_sana_long$figpanel <- 2
# dfLipids_sana_long$figpanel[1:(dim(dfLipids_sana)[1] * floor(dim(dfLipids_sana)[2] / 2))] <- 1

ggplot(data = dfLipids_sana_long, aes(x = variable, y = value)) +
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
dfLipids_sana2 <- dfLipids_sana
dim(dfLipids_sana2)
dfLipids_sana2$Samplingday <- 2
dfLipids_sana2$Samplingday[grep("Base", rownames(dfLipids_sana2))] <- 1
dfLipids_sana2_long <- reshape2::melt(dfLipids_sana2, id.vars = "Samplingday")
nPanels <- 4
nPerPanel <- ceiling(dim(dfLipids_sana2)[2] / nPanels)
dfLipids_sana2_long$figpanel <- 2
SplitsFig <- split(LipidVariableNames_sana, ceiling(seq_along(LipidVariableNames_sana)/nPerPanel))
for (i in 1:length(SplitsFig)) {
  dfLipids_sana2_long$figpanel[which(dfLipids_sana2_long$variable %in% SplitsFig[[i]])] <- i
}
table(dfLipids_sana2_long$figpanel)
ggplot(data = dfLipids_sana2_long, aes(x = variable, y = value, color = factor(Samplingday))) +
  geom_violin(lwd=.2, width = 2, position = position_dodge(1)) +
  #  geom_boxplot(outlier.shape = NA, lwd = .2, position = position_dodge(1)) +
  geom_sina(alpha=0.2, size = .1, position = position_dodge(1)) +
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


# write.csv(dfLipids_sana_scaled_new, "dfLipids_sana_scaled_new")
# write.csv(LipidVariableNames_sana, "LipidVariableNames_sana.csv")
# 
# write.csv(cbind.data.frame(Probe12 = paclitaxel_sana_imputed_log$Probe.1.oder.2,
#                            Neuropaty = paclitaxel_sana_imputed_log$Neuropathie),
#           "dfLipids_Classes_sana.csv")


################################## Group comparison ###################################
names(dfLipids_sana_scaled_new)

t.test(dfLipids_sana_scaled_new$SA1P ~ paclitaxel_sana_imputed_log$Probe.1.oder.2)
t.test(dfLipids_sana_scaled_new$SA1P ~ paclitaxel_sana_imputed_log$Neuropathie..0.5..)

table(paclitaxel_sana_imputed_log$Neuropathie..0.5..)

dfSA1P_12 <- cbind.data.frame(Day = paclitaxel_sana_imputed_log$Probe.1.oder.2, SA1P = paclitaxel_sana_imputed_log$SA1P)
dfSA1P_12_long <- reshape2::melt(dfSA1P_12, id.vars = "Day")


pSana_SA1P_12 <-
  ggplot(dfSA1P_12_long, aes(x = factor(Day), y = value, color = factor(Day), fill = factor(Day))) +
  geom_violin(alpha = .2) +
  geom_boxplot(width = 0.1, alpha = .5, outlier.shape = NA) +
  geom_jitter(width = .1, height = 0) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "t.test", label = "p.format",  label.x.npc = 0.5)+
  theme(legend.position = c(.5,.8), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black'))  +
  labs(color = "Day", x = "Day", y = "log concentration", title = "Sampling day") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")

dfSA1P_NP <- cbind.data.frame(Neuropathy = paclitaxel_sana_imputed_log$Neuropathie..0.5..[paclitaxel_sana_imputed_log$Probe.1.oder.2==2], SA1P = paclitaxel_sana_imputed_log$SA1P[paclitaxel_sana_imputed_log$Probe.1.oder.2==2])
dfSA1P_NP_long <- reshape2::melt(dfSA1P_NP, id.vars = "Neuropathy")


pSana_SA1P_NP <-
  ggplot(dfSA1P_NP_long, aes(x = factor(Neuropathy), y = value, color = factor(Neuropathy), fill = factor(Neuropathy))) +
  geom_violin(alpha = .2) +
  geom_boxplot(width = 0.1, alpha = .5, outlier.shape = NA) +
  geom_jitter(width = .1, height = 0) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "t.test", label = "p.format", label.x.npc = 0.5)+
  theme(legend.position = c(.5,.8), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black'))  +
  labs(color = "Neuropathy", x = "Neuropathy", y = "log concentration", title = "Sampling day") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")

ggarrange(pSana_SA1P_12, pSana_SA1P_NP, labels = paste0(letters[1:2], ")"), ncol = 2,  widths = c(1,1), align = "hv")


plot_grid(HM_uct,
          HM_sana,
          labels = LETTERS[1:5],
  ncol = 2,
  align = "hv",
  axis = "bt"
)
