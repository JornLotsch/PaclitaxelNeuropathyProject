#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 13:52:04 2022

@author: joern
"""

# %% imports

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

from box_and_heatplots import box_and_heatplot
from explore_tukey_lop import explore_tukey_lop
from compute_PCA import perform_pca
from ABCanalysis import ABC_analysis


# %% Read data
# UCT
pfad_o = "/home/joern/Aktuell/PaclitaxelPainLipidomics/"
pfad_u1 = "01Transformierte/sissignano/paclitaxel/results/"
filename = "paclitaxel_uct_imputed.csv"

dfpaclitaxel_uct_imputed = pd.read_csv(pfad_o + pfad_u1 + filename)
dfpaclitaxel_uct_imputed

nonLipidVariableNames = {"Messung", "Patientennummer", "Initialen", "Geburtsjahr", "Probe.1.oder.2", "Zyklus",
                         "Neuropathie..0.5..", "Datum.der.Blutentnahme", "Baehandlungsregime...Dosieung.Paclitaxel..mg.m.2."}
LipidVariableNames = list(
    set(dfpaclitaxel_uct_imputed.columns) - nonLipidVariableNames)

for i, variable in enumerate(LipidVariableNames):
    data_subset = copy.copy(dfpaclitaxel_uct_imputed[variable])
    explore_tukey_lop(data=data_subset)

# %% Transform data
dfpaclitaxel_uct_imputed_log = dfpaclitaxel_uct_imputed.copy()

dfpaclitaxel_uct_imputed_log[LipidVariableNames] = np.log10(
    dfpaclitaxel_uct_imputed_log[LipidVariableNames].astype("float"))
dfpaclitaxel_uct_imputed_log["Neuropathie..0.5.."].fillna(0, inplace=True)
dfpaclitaxel_uct_imputed_log["Neuropathie"] = dfpaclitaxel_uct_imputed_log["Neuropathie..0.5.."]
dfpaclitaxel_uct_imputed_log["Neuropathie"][dfpaclitaxel_uct_imputed_log["Neuropathie"] > 0] = 1

dfpaclitaxel_uct_imputed_log[LipidVariableNames].shape
# %% PCA
PCA_data = dfpaclitaxel_uct_imputed_log[LipidVariableNames].copy()
PCA_data = pd.DataFrame(StandardScaler().fit_transform(
    PCA_data), columns=PCA_data .columns)

y = dfpaclitaxel_uct_imputed_log["Neuropathie"]

PCA_paclitaxel_uct, PCA_paclitaxel_uct_feature_importance = perform_pca(
    PCA_data, target=y, biplot=False, PC_criterion="KaiserGuttman", plotReduced=2)

ABC_A = ABC_analysis(PCA_paclitaxel_uct_feature_importance)
print(list(ABC_A["Aind"].index))

