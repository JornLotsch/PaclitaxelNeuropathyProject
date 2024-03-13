
################################# PCA based clustering ##########################################################

# http://www.sthda.com/english/articles/31-principal-component-methods-in-r-practical-guide/117-hcpc-hierarchical-clustering-on-principal-components-essentials/
library(FactoMineR)
library(factoextra)
library(cluster)
library(dendextend)
library(gridExtra)
library(NbClust)
library(ggthemes)
library(ggpubr)
library(ggmosaic)
library(dplyr)
library(magick)
library(cowplot)

pfad_o0 <- "/home/joern/Aktuell/ProjectionsBiomed/"

source(paste0(pfad_o0, pfad_r, "ProjectionsBiomed_MainFunctions.R"))


# Compute PCA
Lipids.pca <- FactoMineR::PCA(dfLipids, graph = F, ncp = 255)
screeLipids <- fviz_screeplot(Lipids.pca,
  choice = "eigenvalue", ncp = sum(Lipids.pca$eig[, 1] > 1) + 5, addlabels = F, ylim = c(0, 65),
  barcolor = "lemonchiffon4", barfill = "lemonchiffon3"
)
Lipids.pca$eig


PCAvoronoiDay12 <- 
  plotVoronoiTargetProjection3(X = as.data.frame(Lipids.pca$ind$coord[,1:2]), Points = paclitaxel_uct_imputed_log$Probe.1.oder.2) +
  labs(title = "Separation day1/day2")


screeLipids <- screeLipids +
  geom_hline(yintercept = 1, color = "salmon") +
  theme(axis.text.y = element_text(size = 5), axis.text.x = element_text(size = 5)) +
  theme_linedraw()
indLipids <-
  fviz_pca_ind(Lipids.pca,
    col.ind = paclitaxel_uct_imputed$Neuropathie, geom = c("point", "text"), labelsize = 3,
    gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
    repel = TRUE
  ) +
  theme_linedraw() +
  theme(legend.position = "none")

Lipids.pca_contrib <- fviz_contrib(Lipids.pca, choice = "var")
Lipids_contrib <- data.frame(Lipids.pca_contrib$data$name)
for (i in 1:sum(Lipids.pca$eig[, 1] > 1)) {
  con <- fviz_contrib(Lipids.pca, choice = "var", axes = i)
  Lipids_contrib <- cbind.data.frame(Lipids_contrib, con$data$contrib)
}
Lipids_contrib_weighted <- (Lipids_contrib[2:ncol(Lipids_contrib)] / 100) * (Lipids.pca$eig[, 2] / 100)
Lipids_contrib_weighted$Sum <- rowSums(Lipids_contrib_weighted)
ContribSumScore <- cbind.data.frame(Lipids_contrib$Lipids.pca_contrib.data.name, Lipids_contrib_weighted$Sum)

Lipids_contrib_ABC <- ABCanalysis(ContribSumScore$`Lipids_contrib_weighted$Sum`, PlotIt = T)
Lipids_contrib_ABCplot <- ABCplotGG(ContribSumScore$`Lipids_contrib_weighted$Sum`) +
  theme_linedraw() +
  theme(legend.position = c(.9, .5)) +
  ggtitle("ABC plot variable importance")
Lipids_contrib_ABC_nested <- ABCanalysis(ContribSumScore$`Lipids_contrib_weighted$Sum`[ContribSumScore$`Lipids_contrib_weighted$Sum` >= Lipids_contrib_ABC$ABLimit], PlotIt = T)
PCA_selectedFeatues <- ContribSumScore$`Lipids_contrib$Lipids.pca_contrib.data.name`[Lipids_contrib_ABC$Aind]
# write.csv(PCA_selectedFeatues, "PCA_selectedFeatues.csv")

head(ContribSumScore[order(-ContribSumScore$`Lipids_contrib_weighted$Sum`), ], length(Lipids_contrib_ABC$Aind))
head(ContribSumScore$`Lipids_contrib$Lipids.pca_contrib.data.name`[order(-ContribSumScore$`Lipids_contrib_weighted$Sum`)], length(Lipids_contrib_ABC$Aind))

ContribSumScore$Color <- rep("lemonchiffon3", nrow(ContribSumScore))
ContribSumScore$Color[ContribSumScore$`Lipids_contrib_weighted$Sum` >= Lipids_contrib_ABC$ABLimit] <- "burlywood4"
ContribSumScore$Color[ContribSumScore$`Lipids_contrib_weighted$Sum` >= Lipids_contrib_ABC_nested$ABLimit] <- "burlywood4"

Lipids.pca.varimp <- ggplot(data = ContribSumScore, aes(
  x = reorder(ContribSumScore$`Lipids_contrib$Lipids.pca_contrib.data.name`, -ContribSumScore$`Lipids_contrib_weighted$Sum`),
  y = ContribSumScore$`Lipids_contrib_weighted$Sum`
)) +
  geom_bar(stat = "identity", fill = ContribSumScore$Color) +
  theme_linedraw() +
  labs(title = "Variable importance", x = NULL, y = "Score") +
  theme(axis.text.x = element_text(size = 5, angle = 90, vjust = 0.5, hjust = 1))


# Clustering
Lipids.pca <- FactoMineR::PCA(dfLipids, graph = F, ncp = sum(Lipids.pca$eig[, 1] > 1))
Lipids.hcpc <- HCPC(Lipids.pca, consol = T, nb.clust = 2, iter.max = 100, graph = F)

dendLipids <- factoextra::fviz_dend(Lipids.hcpc, show_labels = T, cex = .2, type = "rectangle", k_colors = colorblind_pal()(4), ggtheme = theme_linedraw()) + xlab("Cases")
fmapLipids <- factoextra::fviz_cluster(Lipids.hcpc,
  labelsize = .1, col.ind = paclitaxel_uct_imputed$Neuropathie, repel = F,
  geom = c("point"), main = "Factor map", ellipse.tytpe = "t",
  palette = colorblind_pal()(4), ggtheme = theme_linedraw()
) +
  theme(legend.position = "none")
silclust <- Lipids.hcpc$data.clust$clust # car::recode(Lipids.hcpc$data.clust$clust,"1=2; 2=3;3=1")
silLipids <- factoextra::fviz_silhouette(silhouette(as.numeric(silclust), dist(as.matrix(Lipids.pca$ind$coord))),
  palette = alpha(colorblind_pal()(4), .5), legend = "NA", ggtheme = theme_linedraw(), ylim = c(0, 0.3), alpha = .1
) +
  coord_flip() + ylim(0, 0.4) +
  theme(axis.text.y = element_text(size = 5)) +
  ggtitle(sub(".*\n", "", silLipids$labels$title))

fisher.test(table(silclust, paclitaxel_uct_imputed_log$Probe.1.oder.2))
ftNP <- fisher.test(table(silclust, paclitaxel_uct_imputed_log$Neuropathie))

cls_np <- paclitaxel_uct_imputed_log$Neuropathie
mosaicLipids <-
  ggplot(data = cbind.data.frame(silclust, cls_np)) +
  geom_mosaic(aes(x = product(silclust, cls_np), fill = silclust), na.rm = TRUE, alpha = .6) +
  scale_fill_colorblind() +
  theme_linedraw() +
  theme(legend.position = "none") +
  annotate(geom = "text", label = paste0("Fisher test: odds ratio = ", round(ftNP$estimate, 3), "\np-value = ", round(ftNP$p.value, 3)), x = .5, y = .5) +
  labs(title = "Contingency table cluster versus neuropathy", x = "Neuropathy", y = "Cluster")


# Plot results
ggarrange(ggarrange(
  ggarrange(indLipids, labels = paste0(letters[1], ")")),
  ggarrange(screeLipids, mosaicLipids, fmapLipids, dendLipids,
    labels = paste0(letters[2:5], ")"),
    ncol = 2, nrow = 2, align = "hv"
  )
),
ggarrange(Lipids.pca.varimp, Lipids_contrib_ABCplot, widths = c(3, 1), labels = paste0(letters[6:7], ")")),
ncol = 1, heights = c(2, 1), align = "hv"
)


# without clusters

image_Umx_2 <- image_read(paste0(pfad_o, "05OurPublication/Bilder/", "Lipids_Umx_3d_2.png"))

# 1400 x 2400
plot_grid(plot_grid(
  plot_grid(indLipids, labels = paste0(letters[1], ")")),
  plot_grid(PCAvoronoiDay12, Lipids_contrib_ABCplot,
            labels = paste0(letters[2:3], ")"), 
            ncol = 1, nrow = 2, align = "hv"
  ),
  rel_widths = c(2,1)
),
plot_grid(Lipids.pca.varimp, 
          image_ggplot(image_Umx_2) + ggtitle("U-matrix"),
          ncol = 1, algin = "v", axes = "lr",
          labels = paste0(letters[4:5], ")"),
          rel_heights  = c(4, 2)),
ncol = 1, rel_heights  = c(2, 3), align = "hv"
)





# Cluster stability
library(fossil)
library(psych)
library(cluster)
library(clValid)
library(clusterSim)
library(parallel)
library(pbmcapply)

num_workers <- parallel::detectCores()
nProc <- num_workers - 1

nIter <- 100
Daten <- cbind(silclust, dfLipids)
ClusterAktualPreset <- pbmcapply::pbmclapply(1:nIter, function(i) {
  set.seed(42 + i)
  Daten_i <- Daten[sample(nrow(Daten), nrow(Daten), replace = T), ]
  res.pca.Daten <- FactoMineR::PCA(subset(Daten_i, select = LipidVariableNames), graph = F, ncp = sum(Lipids.pca$eig[, 1] > 1))
  res.hcpc.Daten <- FactoMineR::HCPC(res.pca.Daten, graph = F, nb.clust = 2)
  ClusterAktual <- as.numeric(res.hcpc.Daten$data.clust$clust)
  ClusterPreset <- as.numeric(Daten_i$silclust)
  return(list(ClusterAktual = ClusterAktual, ClusterPreset = ClusterPreset))
}, mc.cores = nProc)

randIndex <- vector()
for (i in 1:nIter)
{
  randIndex <- append(randIndex, adj.rand.index(ClusterAktualPreset[[i]][["ClusterAktual"]], ClusterAktualPreset[[i]][["ClusterPreset"]]))
}
describe(randIndex)
quantile(randIndex, probs = c(0.025, 0.5, 0.975), na.rm = T)

ClusterQuality <- pbmcapply::pbmclapply(1:nIter, function(i) {
  set.seed(42 + i)
  Daten_i <- Daten[sample(nrow(Daten), nrow(Daten), replace = T), ]
  res.pca.Daten <- FactoMineR::PCA(subset(Daten_i, select = LipidVariableNames), graph = F, ncp = sum(Lipids.pca$eig[, 1] > 1))
  res.hcpc.Daten <- FactoMineR::HCPC(res.pca.Daten, graph = F, nb.clust = 2)
  d_euc <- stats::dist(as.matrix(res.pca.Daten$ind$coord), method = "euclidian")
  SilhouetteIndexMean <- mean(silhouette(as.numeric(res.hcpc.Daten$data.clust$clust), dist(as.matrix(res.pca.Daten$ind$coord[, 1:2])))[, 3])
  DunnIndex <- clValid::dunn(distance = dist(as.matrix(res.pca.Daten$ind$coord[, 1:2])), clusters = as.numeric(res.hcpc.Daten$data.clust$clust))
  clu2 <- pam(as.matrix(res.pca.Daten$ind$coord[, 1:2]), 3)
  DaviesBouldinIndex <- DaviesBouldin(as.matrix(res.pca.Daten$ind$coord[, 1:2]), as.numeric(res.hcpc.Daten$data.clust$clust))$dBIndex
  return(list(SilhouetteIndexMean = SilhouetteIndexMean, DunnIndex = DunnIndex, DaviesBouldinIndex = DaviesBouldinIndex))
}, mc.cores = nProc)

unlist(lapply(lapply(ClusterQuality, "[[", "SilhouetteIndexMean"), "[[", 1))
describe(unlist(lapply(lapply(ClusterQuality, "[[", "SilhouetteIndexMean"), "[[", 1)))
quantile(unlist(lapply(lapply(ClusterQuality, "[[", "SilhouetteIndexMean"), "[[", 1)), probs = c(0.025, 0.5, 0.975), na.rm = T)
quantile(unlist(lapply(lapply(ClusterQuality, "[[", "DunnIndex"), "[[", 1)), probs = c(0.025, 0.5, 0.975), na.rm = T)
quantile(unlist(lapply(lapply(ClusterQuality, "[[", "DaviesBouldinIndex"), "[[", 1)), probs = c(0.025, 0.5, 0.975), na.rm = T)

########## U matrix #############################################
pfad_umx <- "/08AnalyseProgramme/PaclitaxelNeuropathyProject/R"

library(Umatrix)
Cls_Umx <- paclitaxel_uct_imputed$Neuropathie
Cls_Umx[is.na(Cls_Umx)] <- 3
# Lipids_Umx <- Umatrix::iEsomTrain(Data = dfLipids_scaled, Cls = Cls_Umx)
# WriteBM(FileName = "Lipids_Umx.bm", BestMatches = Lipids_Umx$BestMatches)
# WriteUMX(FileName = "Lipids_Umx.umx", UMatrix = Lipids_Umx$Umatrix)
# WriteWTS(FileName = "Lipids_Umx.wts", wts = Lipids_Umx$Weights)

BestMatches <- ReadBM(FileName = "Lipids_Umx.bm", InDirectory = paste0(pfad_o, pfad_umx))
UMatrix <- ReadUMX(FileName = "Lipids_Umx.umx", InDirectory = paste0(pfad_o, pfad_umx))
Weigths <- ReadWTS(FileName = "Lipids_Umx.wts", InDirectory = paste0(pfad_o, pfad_umx))

# Lipids_Imx <- Umatrix::iUmapIsland(Umatrix = UMatrix, BestMatches = BestMatches$BestMatches, Cls = Cls_Umx)
# WriteIMX("Lipids_Imx.imx", Lipids_Imx$Imx)
Imx <- ReadIMX("Lipids_Imx.imx", InDirectory = paste0(pfad_o, pfad_umx))

# Lipids_Cls <- Umatrix::iClassification(Umatrix = UMatrix, BestMatches = BestMatches$BestMatches, Cls = Cls_Umx, Imx = Imx)
# WriteCLS("Lipids_Cls.cls", Lipids_Cls$Cls)
Lipids_ClsUmx <- ReadCLS("Lipids_Cls.cls", InDirectory = paste0(pfad_o, pfad_umx))
Lipids_ClsUmx$Cls[Lipids_ClsUmx$Cls > 1] <- 2
table(Lipids_ClsUmx$Cls)

Umatrix::plotMatrix(
  Matrix = UMatrix, BestMatches = BestMatches$BestMatches, Cls = Lipids_ClsUmx$Cls, Imx = Imx,
  TransparentContours = T, BmSize = 10, RemoveOcean = T
)
Umatrix::showMatrix3D(
  Matrix = UMatrix, BestMatches = BestMatches$BestMatches, Cls = Lipids_ClsUmx$Cls, Imx = Imx,
  BmSize = 1, RemoveOcean = T
)
library(rgl)
snapshot3d("Lipids_Umx_3d_2.png", "png")


fUmsD1D2 <- fisher.test(table(Lipids_ClsUmx$Cls, paclitaxel_uct_imputed$Probe.1.oder.2))
fisher.test(table(Lipids_ClsUmx$Cls, paclitaxel_uct_imputed$Neuropathie))
fisher.test(table(Lipids_ClsUmx$Cls, silclust))

# write.csv(cbind.data.frame(Probe12 = paclitaxel_uct_imputed_log$Probe.1.oder.2,
#                            Neuropaty = paclitaxel_uct_imputed_log$Neuropathie,
#                            Clusters = silclust), "dfLipids_Classes.csv")
# write.csv(rownames(paclitaxel_uct_imputed_log), "Index.csv")



mosaicUmxD1D2 <-
  ggplot(data = cbind.data.frame(Umx = Lipids_ClsUmx$Cls, Original = paclitaxel_uct_imputed$Probe.1.oder.2)) +
  geom_mosaic(aes(x = product(Umx, Original), fill = Umx), na.rm = TRUE, alpha = .6) +
  scale_fill_colorblind() +
  theme_linedraw() +
  theme(legend.position = "none") +
  annotate(geom = "text", label = paste0("Fisher test: odds ratio = ", round(fUmsD1D2$estimate, 3), "\np-value = ", round(fUmsD1D2$p.value, 3)), x = .5, y = .5, color = "white") +
  labs(title = "Contingency table U-matrix versus day1/2", x = "Day1/2", y = "Umatrix-Clusters") +
  coord_fixed()

  

image_Umx <- image_read(paste0(pfad_o, "05OurPublication/Bilder/", "Lipids_Umx_3d.png"))

# 1400 x 2400
plot_grid( image_ggplot(image_Umx) + ggtitle("U-matrix"),
           mosaicUmxD1D2,
          nrow = 1, 
          #algin = "h", axes = "tb",
          labels = paste0(letters[1:2], ")"),
          rel_widths  = c(2, 1)
          )

