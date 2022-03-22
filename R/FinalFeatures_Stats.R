#View(paclitaxel_uct_imputed_log)
names(paclitaxel_uct_imputed_log)

Lipids_Final <- subset(paclitaxel_uct_imputed_log, select = c("Patientennummer", "Probe.1.oder.2","Neuropathie", 
                                                              "SA1P", "Sphingomyelin_331", "Sphingomyelin_431"))

PatNrNeuropathy <- na.omit(Lipids_Final$Patientennummer[Lipids_Final$Neuropathie == 1])
Lipids_Final$Neuropathie1 <- Lipids_Final$Neuropathie
Lipids_Final$Neuropathie1[is.na(Lipids_Final$Neuropathie1)] <- 0
Lipids_Final$Neuropathie1[Lipids_Final$Patientennummer %in% PatNrNeuropathy] <- 1

Lipids_Final$Day123 <- Lipids_Final$Probe.1.oder.2 + Lipids_Final$Neuropathie1
Lipids_Final$Day124 <- 2 * Lipids_Final$Probe.1.oder.2 + Lipids_Final$Neuropathie1
table(Lipids_Final$Day124)

Lipids_Final_long <- reshape2::melt(Lipids_Final, id.vars = c("Patientennummer", "Day123", "Day124","Probe.1.oder.2","Neuropathie","Neuropathie1"))

library(ggplot2)
library(ggpubr)
library(ggthemes)

ggplot(data = Lipids_Final_long, aes(x = factor(Day123), y = value, color = factor(Day123))) +
  geom_violin() +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  geom_jitter(width = .1) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() 

ggplot(data = Lipids_Final_long, aes(x = factor(Day124), y = value, color = factor(Day124))) +
  geom_violin() +
  geom_boxplot(width = 0.2, outlier.shape = NA) +
  geom_jitter(width = .1, height = 0) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "kruskal.test") +
  theme(legend.position = c(.25,.8)) +
  labs(color = "Group", x = "Group") +
  scale_color_manual(values = colorblind_pal()(8)[2:6])

plot_lipids_day <- ggplot(data = Lipids_Final_long[Lipids_Final_long$variable == "SA1P",] , aes(x = factor(Probe.1.oder.2), y = value, color = factor(Probe.1.oder.2), fill = factor(Probe.1.oder.2))) +
  geom_violin(alpha = .2) +
  geom_boxplot(width = 0.1, alpha = .5, outlier.shape = NA) +
  geom_jitter(width = .1, height = 0) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "kruskal.test", label = "p.format")+
  theme(legend.position = c(.8,.8), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black'))  +
  labs(color = "Day", x = "Day", y = "log concentration", title = "Sampling day") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")

plot_lipids_neuropathy <- ggplot(data = Lipids_Final_long[Lipids_Final_long$Probe.1.oder.2 == 2,], aes(x = factor(Neuropathie), y = value, color = factor(Neuropathie), fill = factor(Probe.1.oder.2))) +
  geom_violin(alpha = .2) +
  geom_boxplot(width = 0.1, alpha = .5, outlier.shape = NA) +
  geom_jitter(width = .1) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "kruskal.test", label = "p.format")+
  theme(legend.position = c(.1,.85), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black')) +
  labs(color = "Neuropathy", x = "Neuropathy", y = "log concentration", title = "Neuropathy - related lipid markers:  day 2, neuopathy yes/no") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")
# Plot results
#ggarrange(plot_lipids_day, plot_lipids_neuropathy, labels = paste0(letters[1:2], ")"), ncol = 1,  align = "hv")
ggarrange(plot_lipids_day, plot_lipids_neuropathy, labels = paste0(letters[1:2], ")"), ncol = 2,  widths = c(1,3), align = "hv")


# Test bayes classifier
Means_SA1P <- c(mean(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==0])) ,mean(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==1])) )
SDs_SA1P <- c(sd(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==0])) ,sd(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==1])) )
Weights_SA1P <- c(length(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==0])) / 
                    length(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie>=0])) ,length(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie==1])) / 
                    length(na.omit(Lipids_Final$SA1P[Lipids_Final$Neuropathie>=0]))  )

LimitBayes <- BayesDecisionBoundaries(Means = Means_SA1P, SDs = SDs_SA1P, Weights = Weights_SA1P)
NewNeuropathy <- ifelse(Lipids_Final$SA1P < LimitBayes, 0 ,1 )
NewNeuropathy[which(is.na(Lipids_Final$Neuropathie))] <- NA

library(caret)
confusionMatrix(factor(NewNeuropathy), factor(Lipids_Final$Neuropathie))




