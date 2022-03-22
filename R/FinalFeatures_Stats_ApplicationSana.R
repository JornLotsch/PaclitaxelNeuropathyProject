#View(paclitaxel_sana_imputed_log)
names(paclitaxel_sana_imputed_log)

Lipids_Final_sana <- subset(paclitaxel_sana_imputed_log, select = c("Patientennummer", "Probe.1.oder.2","Neuropathie", 
                                                              "SA1P", "Sphingomyelin_331", "Sphingomyelin_431"))

paclitaxel_sana_imputed_log$Sphingomyelin_331

PatNrNeuropathy <- na.omit(Lipids_Final_sana$Patientennummer[Lipids_Final_sana$Neuropathie == 1])
Lipids_Final_sana$Neuropathie1 <- Lipids_Final_sana$Neuropathie
Lipids_Final_sana$Neuropathie1[is.na(Lipids_Final_sana$Neuropathie1)] <- 0
Lipids_Final_sana$Neuropathie1[Lipids_Final_sana$Patientennummer %in% PatNrNeuropathy] <- 1

Lipids_Final_sana$Day123 <- Lipids_Final_sana$Probe.1.oder.2 + Lipids_Final_sana$Neuropathie1
Lipids_Final_sana$Day124 <- 2 * Lipids_Final_sana$Probe.1.oder.2 + Lipids_Final_sana$Neuropathie1
table(Lipids_Final_sana$Day124)

Lipids_Final_long <- reshape2::melt(Lipids_Final_sana, id.vars = c("Patientennummer", "Day123", "Day124","Probe.1.oder.2","Neuropathie","Neuropathie1"))

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
  stat_compare_means(method = "wilcox.test")+
  theme(legend.position = c(.25,.8), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black'))  +
  labs(color = "Day", x = "Day", y = "log concentration", title = "Neuropathy - related lipid markers:  Sampling day") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")

plot_lipids_neuropathy <- ggplot(data = Lipids_Final_long[Lipids_Final_long$Probe.1.oder.2 == 2,], aes(x = factor(Neuropathie), y = value, color = factor(Neuropathie), fill = factor(Probe.1.oder.2))) +
  geom_violin(alpha = .2) +
  geom_boxplot(width = 0.1, alpha = .5, outlier.shape = NA) +
  geom_jitter(width = .1) +
  facet_wrap(~ variable, scale = "free") +
  theme_linedraw() +
  stat_compare_means(method = "wilcox.test")+
  theme(legend.position = c(.1,.85), strip.background = element_rect(fill="cornsilk"), strip.text = element_text(colour = 'black')) +
  labs(color = "Neuropathy", x = "Neuropathy", y = "log concentration", title = "Neuropathy - related lipid markers:  day 2, neuopathy yes/no") +
  scale_color_manual(values = c("antiquewhite4", "dodgerblue4")) +
  scale_fill_manual(values = c("antiquewhite4", "dodgerblue4"), guide = "none")
# Plot results
ggarrange(plot_lipids_day, plot_lipids_neuropathy, labels = paste0(letters[1:2], ")"), ncol = 1,  align = "hv")

