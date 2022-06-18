#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""
#https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-feature-selection

# %% imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import SelectFromModel, RFE, SequentialFeatureSelector, SelectKBest, f_classif, SelectFpr, SelectFwe
from sklearn.ensemble import RandomForestClassifier
from ABCanalysis import ABC_analysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV


# %% Functions
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return (u1 - u2) / s

def annotate_axes(ax, text, fontsize=18):
    ax.text(-.021, 1.0, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black")


# %% Read scaled data and classes from R
pfad_o1 = "/home/joern/Aktuell/PaclitaxelPainLipidomics/"
pfad_u3 = "08AnalyseProgramme/R/"

dfPaclitaxel = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_scaled.csv")
dfPaclitaxel.columns
dfPaclitaxel.set_index("Unnamed: 0", inplace=True)
dfPaclitaxel.columns
dfPaclitaxel.index

Paclitaxel_targets = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_Classes.csv")
Paclitaxel_targets .columns


# %% Cluster explnation ML

y = Paclitaxel_targets["Probe12"]
pd.DataFrame(y).value_counts()


X_train, X_test, y_train, y_test = train_test_split(
    dfPaclitaxel, y, test_size=0.2, random_state=42)
pd.DataFrame(y_test).value_counts()
# https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6

#%% Classifier tuning 

# LinearSVC
lsvc = LinearSVC(max_iter=10000)
param_grid = {"C": np.arange(0.01,100,10), 
              "penalty": ["l1", "l2"], 
              "dual": [True, False], 
              "loss": ["hinge", "squared_hinge"],
              "tol": [0.001,0.0001,0.00001]}
grid_search = GridSearchCV(lsvc, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
C_lsvm, dual_svm, loss_svm, penalty_SVM, tol_svm = grid_search.best_params_.values()

# SVM SVC
svc = SVC(max_iter=10000,random_state=0)
param_grid = {"C": np.arange(0.01,100,10), 
              "kernel": ["linear", "poly", "rbf", "sigmoid"], 
              "gamma": [1e-3, 1e-4, "scale", "auto"],
              "tol": [0.001,0.0001,0.00001]}
grid_search = GridSearchCV(svc, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
C_svs, gamma_svs, kernel_svs, tol_svs = grid_search.best_params_.values()
    
# Random forests
forest = RandomForestClassifier(random_state=0)
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
grid_search = GridSearchCV(forest, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
bootstrap_rf,  max_depth_rf, max_features_rf, min_samples_leaf_rf, min_samples_split_rf, n_estimators_rf = grid_search.best_params_.values()

# Logistic regregssion
LogReg = LogisticRegression(max_iter=10000, random_state=0)
param_grid ={"C":np.logspace(-3,3,7),
             "tol": [0.001,0.0001,0.00001],
             "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
grid_search = GridSearchCV(LogReg, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
C_LogReg,  solver_LogReg, tol_LogReg= grid_search.best_params_.values()
penalty_LogReg = "l2"

# kNN
kNN = KNeighborsClassifier()
param_grid ={"n_neighbors":[3,5,7,9],
             "p": [1,2,3],
             "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
             "leaf_size": [10, 20, 30, 40, 50],
             "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobi"]}
grid_search = GridSearchCV(kNN, param_grid=param_grid, scoring="balanced_accuracy", verbose=0, n_jobs = -1)
grid_search.fit(X_train, y_train)
algorithm_kNN, leaf_size_kNN, metric_kNN, n_neighbors_kNN, p_kNN, = grid_search.best_params_.values()


# %% Create results table all methods and variables
featureSelection_methods = ["PCA", "features_CohenD","features_FPR", "features_FWE", 
                            "features_lSVC_sKb", "features_RF_sKb", "features_LogReg_sKb", 
                            "features_lSVC_sfm", "features_RF_sfm", "features_LogReg_sfm",
                            "features_lSVC_rfe", "features_RF_rfe", "features_LogReg_rfe",
                            "features_lSVC_sfs_forward", "features_RF_sfs_forward", "features_LogReg_sfs_forward",
                            "features_lSVC_sfs_backward", "features_RF_sfs_backward", "features_LogReg_sfs_backward"]
                            
feature_table = pd.DataFrame(np.zeros((len(X_train.columns),len(featureSelection_methods))))
feature_table.columns = featureSelection_methods
feature_table.set_index(X_train.columns, inplace=True)

# Add PCA results from previous anaylsis

PCA_selectedFeatures = pd.read_csv(pfad_o1 + pfad_u3 + "PCA_selectedFeatues.csv")
PCA_selectedFeatures .columns

feature_table.loc[PCA_selectedFeatures["x"].tolist(), "PCA"] = 1
                       
# %% CV
n_splits = 5
n_repeats = 20

#%% Cohens's d CV

features_CohenD = []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    X_train_FS_CohenD = X_train_FS.copy()
    X_train_FS_CohenD["y"] = list(y_train_FS)
    
    chd = [] 
    for i1 in range(X_train_FS_CohenD.shape[1]-1):
        chd .append(abs(cohend(*[group[X_train_FS_CohenD.columns[i1]].values for name, group in X_train_FS_CohenD.groupby(X_train_FS_CohenD.iloc[:, -1])])))
    
    
    df_chd= pd.DataFrame(X_train_FS.columns, columns = ["variable"])
    df_chd.set_index("variable",inplace=True) 

    df_chd["Cohens' d"] = chd
    features_CohenD.append(ABC_analysis(data = df_chd["Cohens' d"])["Aind"].index)
    
features_CohenD_all = []
for i in range(len(features_CohenD)):
    for j in range(len(features_CohenD[i])):
        features_CohenD_all.append(features_CohenD[i][j])
features_CohenD_all = pd.DataFrame({"Counts":  pd.DataFrame(features_CohenD_all).value_counts()})
features_CohenD_all.reset_index()
ABCres = ABC_analysis(features_CohenD_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_CohenD"] = 1

fig, ax = plt.subplots(figsize=(18, 16))
sns.barplot(ax=ax, x=features_CohenD_all.index.to_numpy(), y=features_CohenD_all["Counts"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% FPR based selection

features_FPR = []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    select_fpr_classif = SelectFpr(score_func=f_classif)
    select_fpr_classif.fit(X_train_FS, y_train_FS)

    feature_idx = select_fpr_classif.get_support()
    feature_name = X_train.columns[feature_idx]
    
    features_FPR.append(feature_name)
    
features_FPR_all = []
for i in range(len(features_FPR)):
    for j in range(len(features_FPR[i])):
        features_FPR_all.append(features_FPR[i][j])
features_FPR_all = pd.DataFrame({"Counts":  pd.DataFrame(features_FPR_all).value_counts()})
features_FPR_all.reset_index()
ABCres = ABC_analysis(features_FPR_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_FPR"] = 1

fig, ax = plt.subplots(figsize=(18, 16))
sns.barplot(ax=ax, x=features_FPR_all.index.to_numpy(), y=features_FPR_all["Counts"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% FWE based selection

features_FWE = []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    select_fwe_classif = SelectFwe(score_func=f_classif)
    select_fwe_classif.fit(X_train_FS, y_train_FS)

    feature_idx = select_fwe_classif.get_support()
    feature_name = X_train.columns[feature_idx]
    
    features_FWE.append(feature_name)
    
features_FWE_all = []
for i in range(len(features_FWE)):
    for j in range(len(features_FWE[i])):
        features_FWE_all.append(features_FWE[i][j])
features_FWE_all = pd.DataFrame({"Counts":  pd.DataFrame(features_FWE_all).value_counts()})
features_FWE_all.reset_index()
ABCres = ABC_analysis(features_FWE_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_FWE"] = 1

fig, ax = plt.subplots(figsize=(18, 16))
sns.barplot(ax=ax, x=features_FWE_all.index.to_numpy(), y=features_FWE_all["Counts"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% Feature selection univariate selectKbest

features_lSVC_sKb = []
features_RF_sKb = []
features_LogReg_sKb = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]
    
    
    anova_filter = SelectKBest(f_classif, k=3)
    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    #lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("lsvc", lsvc)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
        )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_lSVC_sKb.append(X_new.get_feature_names_out().tolist())

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf, n_jobs = -1)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("forest", forest)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
    )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_RF_sKb.append(X_new.get_feature_names_out().tolist())

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0, n_jobs = -1)
    pipeline = Pipeline([("anova_filter", anova_filter ), ("LogReg", LogReg)])
    param_grid = dict(
        anova_filter__k=list(range(1,X_train_FS.shape[1])),
    )
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="balanced_accuracy", verbose=0,n_jobs = -1)
    grid_search.fit(X_train_FS, y_train_FS)
    k = max(grid_search.best_params_.values())

    X_new = SelectKBest(f_classif, k = k).fit(X_train_FS, y_train_FS)
    features_LogReg_sKb.append(X_new.get_feature_names_out().tolist())
    
features_lSVC_sKb_all = []
for i in range(len(features_lSVC_sKb)):
    for j in range(len(features_lSVC_sKb[i])):
        features_lSVC_sKb_all.append(features_lSVC_sKb[i][j])
features_lSVC_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sKb_all).value_counts()})
features_lSVC_sKb_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sKb"] = 1

features_RF_sKb_all = []
for i in range(len(features_RF_sKb)):
    for j in range(len(features_RF_sKb[i])):
        features_RF_sKb_all.append(features_RF_sKb[i][j])
features_RF_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sKb_all).value_counts()})
features_RF_sKb_all.reset_index()
ABCres = ABC_analysis(features_RF_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sKb"] = 1

features_LogReg_sKb_all = []
for i in range(len(features_LogReg_sKb)):
    for j in range(len(features_LogReg_sKb[i])):
        features_LogReg_sKb_all.append(features_LogReg_sKb[i][j])
features_LogReg_sKb_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sKb_all).value_counts()})
features_LogReg_sKb_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sKb_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sKb"] = 1

features_sKb = pd.concat({"SVM": features_lSVC_sKb_all,
                         "RF": features_RF_sKb_all, "LogReg": features_LogReg_sKb_all}, axis=1)
variablenames = []
for i in range(len(features_sKb .index)):
    variablenames.append(features_sKb .index[i][0])
features_sKb["variable"] = (variablenames)
features_sKb.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
sns.barplot(ax=ax, x=features_sKb.index.tolist(), y=features_sKb.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% Feature selection Select from model

features_lSVC_sfm = []
features_RF_sfm = []
features_LogReg_sfm = []
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000).fit(
        X_train_FS, y_train_FS) 
    # lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0).fit(
    #     X_train_FS, y_train_FS)
    model_lsvc = SelectFromModel(lsvc, prefit=True)
    feature_idx = model_lsvc.get_support()
    feature_name = X_train.columns[feature_idx]
    features_lSVC_sfm.append(feature_name)

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf, n_jobs = -1).fit(X_train_FS, y_train_FS)
    model_forest = SelectFromModel(forest, prefit=True)
    feature_idx = model_forest.get_support()
    feature_name = X_train.columns[feature_idx]
    features_RF_sfm.append(feature_name)

    LogReg = LogisticRegression(
        C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0, n_jobs = -1).fit(X_train_FS, y_train_FS)
    model_reg = SelectFromModel(LogReg, prefit=True)
    feature_idx = model_reg.get_support()
    feature_name = X_train.columns[feature_idx]
    features_LogReg_sfm.append(feature_name)

features_lSVC_sfm_all = []
for i in range(len(features_lSVC_sfm)):
    for j in range(len(features_lSVC_sfm[i])):
        features_lSVC_sfm_all.append(features_lSVC_sfm[i][j])
features_lSVC_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_lSVC_sfm_all).value_counts()})
features_lSVC_sfm_all.reset_index()
ABCres = ABC_analysis(features_lSVC_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_lSVC_sfm"] = 1

features_RF_sfm_all = []
for i in range(len(features_RF_sfm)):
    for j in range(len(features_RF_sfm[i])):
        features_RF_sfm_all.append(features_RF_sfm[i][j])
features_RF_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_RF_sfm_all).value_counts()})
features_RF_sfm_all.reset_index()
ABCres = ABC_analysis(features_RF_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_RF_sfm"] = 1

features_LogReg_sfm_all = []
for i in range(len(features_LogReg_sfm)):
    for j in range(len(features_LogReg_sfm[i])):
        features_LogReg_sfm_all.append(features_LogReg_sfm[i][j])
features_LogReg_sfm_all = pd.DataFrame({"Counts":  pd.DataFrame(features_LogReg_sfm_all).value_counts()})
features_LogReg_sfm_all.reset_index()
ABCres = ABC_analysis(features_LogReg_sfm_all.iloc[:,0])
ABCres["Aind"].reset_index(inplace=True)
feature_table.loc[ABCres["Aind"].iloc[:,0].tolist(), "features_LogReg_sfm"] = 1

features_SFM = pd.concat({"SVM": features_lSVC_sfm_all,
                         "RF": features_RF_sfm_all, "LogReg": features_LogReg_sfm_all}, axis=1)
variablenames = []
for i in range(len(features_SFM .index)):
    variablenames.append(features_SFM .index[i][0])
features_SFM["variable"] = (variablenames)
features_SFM.set_index("variable", inplace=True)

fig, ax = plt.subplots(figsize=(18, 16))
sns.barplot(ax=ax, x=features_SFM.index.tolist(), y=features_SFM.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% Feature selection RFE

i = 0
features_lSVC_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
features_RF_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
features_LogReg_rfe = pd.DataFrame(np.zeros(
    (len(X_train.columns), n_splits*n_repeats)), index=X_train.columns.tolist())
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    # lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    rfe = RFE(estimator=lsvc, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_lSVC_rfe.iloc[:, i] = ranking

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    rfe = RFE(estimator=forest, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_RF_rfe.iloc[:, i] = ranking

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    rfe = RFE(estimator=LogReg, n_features_to_select=1, step=1)
    rfe.fit(X_train_FS, y_train_FS)
    ranking = rfe.ranking_
    ranking = max(ranking) - ranking
    features_LogReg_rfe.iloc[:, i] = ranking

    i += 1

features_lSVC_rfe_all = features_lSVC_rfe.sum(
    axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_lSVC_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_lSVC_rfe"] = 1

features_RF_rfe_all = features_RF_rfe.sum(axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_RF_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_RF_rfe"] = 1

features_LogReg_rfe_all = features_LogReg_rfe.sum(
    axis=1).sort_values(ascending=False)
ABCres = ABC_analysis(features_LogReg_rfe_all)
feature_table.loc[ABCres["Aind"].index.tolist(), "features_LogReg_rfe"] = 1

features_RFE = pd.concat({"SVM": features_lSVC_rfe_all,
                         "RF": features_RF_rfe_all, "LogReg": features_LogReg_rfe_all}, axis=1)

fig, ax = plt.subplots(figsize=(18, 16))
ax = sns.barplot(x=features_RFE.index.tolist(), y=features_RFE.iloc[:, 0])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)


# %% Save results in table
feature_table_day12 = feature_table.copy()
file = "feature_table_day12.csv"
feature_table_day12.to_csv(file)

# %% feature selection sum score for ABC for selection of final feature set

FS_sumscore_day12 = feature_table_day12.sum(axis = 1)
FS_sumscore_day12.sort_values(ascending = False, inplace=True)

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(12, 14))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, wspace=.1, hspace=0.4)

    ax2 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    axes = [ax2, ax1]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

    ABC_A_FS_sumscore = ABC_analysis(
        ax=ax1, data=FS_sumscore_day12, PlotIt=True)
    ABC_A_FS_sumscore_nested = ABC_analysis(ABC_A_FS_sumscore["Aind"]["value"])

    barcols = ["wheat" if (i) < ABC_A_FS_sumscore["ABlimit"] else "tan" if i <
               ABC_A_FS_sumscore_nested["ABlimit"] else "tan" for i in FS_sumscore_day12]
    ax1.set_title("ABC plot")
    sns.barplot(ax=ax2, x=FS_sumscore_day12.index.tolist(),
                y=FS_sumscore_day12, palette=barcols, alpha=1)
    ax2.set_title("Number of selections by 17 different methods")
    ax2.set_xlabel(None)
    ax2.set_ylabel("Times selected")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

# with sns.axes_style("darkgrid"):
fig = plt.figure(figsize=(32, 14))
gs0 = gridspec.GridSpec(2, 2, figure=fig, wspace=.1, hspace=0.4)

ax2 = fig.add_subplot(gs0[1, :2])
ax1 = fig.add_subplot(gs0[0, 0])
ax3 = fig.add_subplot(gs0[0, 1])
axes = [ax2, ax1, ax3]
for i, ax in enumerate(axes):
    annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

ABC_A_FS_sumscore = ABC_analysis(
    ax=ax1, data=FS_sumscore_day12, PlotIt=True)
ABC_A_FS_sumscore_nested = ABC_analysis(ax=ax3, data = ABC_A_FS_sumscore["Aind"]["value"], PlotIt=True)

barcols = ["wheat" if (i) < ABC_A_FS_sumscore["ABlimit"] else "peru" if i <
           ABC_A_FS_sumscore_nested["ABlimit"] else "saddlebrown" for i in FS_sumscore_day12]
ax1.set_title("ABC plot")
ax3.set_title("ABC plot (nested)")
sns.barplot(ax=ax2, x=FS_sumscore_day12.index.tolist(),
            y=FS_sumscore_day12, palette=barcols, alpha=1)
ax2.set_title("Number of selections by 17 different methods")
ax2.set_xlabel(None)
ax2.set_ylabel("Times selected")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

# %% Final features

reduced_feature_names = ABC_A_FS_sumscore["Aind"].index.tolist()
sparse_feature_names = ABC_A_FS_sumscore_nested["Aind"].index.tolist()

file = "reduced_feature_names_day12.csv"
ABC_A_FS_sumscore["Aind"].to_csv(file)
file = "sparse_feature_names_day12.csv"
ABC_A_FS_sumscore_nested["Aind"].to_csv(file)

#%% Test of selected features whether they suffice to predict the separate validation data subset
# Balanced accuracy and ROC AUC

BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet, BA_kNN_fullFeatureSet  = [], [], [], []
BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet , BA_kNN_reducedFeatureSet = [], [], [], []
BA_lsvc_reducedFeatureSet_permuted, BA_RF_reducedFeatureSet_permuted, BA_LogReg_reducedFeatureSet_permuted , BA_kNN_reducedFeatureSet_permuted = [], [], [], []
BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet , BA_kNN_sparseFeatureSet = [], [], [], []
ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet, ROC_kNN_fullFeatureSet = [], [], [], []
ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet, ROC_kNN_reducedFeatureSet  = [], [], [], []
ROC_lsvc_reducedFeatureSet_permuted, ROC_RF_reducedFeatureSet_permuted, ROC_LogReg_reducedFeatureSet_permuted, ROC_kNN_reducedFeatureSet_permuted  = [], [], [], []
ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet , ROC_kNN_sparseFeatureSet = [], [], [], []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]
    
    X_train_validation, X_test_validation, y_train_validation, y_test_validation = train_test_split(
        X_test, y_test, test_size=0.8)
    y_test_validation_num = [0 if x == 1 else 1 for x in y_test_validation]

    y_test_validation_permuted = y_test_validation.copy()
    y_test_validation_permuted = pd.Series(np.random.permutation(y_test_validation_permuted))
    y_test_validation_num_permuted = y_test_validation_num.copy()
    y_test_validation_num_permuted = pd.Series(np.random.permutation(y_test_validation_num_permuted))

    # Full feature set
    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    lsvc.fit(X_train_FS, y_train_FS)
    clf = CalibratedClassifierCV(lsvc) 
    clf.fit(X_train_FS, y_train_FS)
    y_pred = lsvc.predict(X_test_validation)
    y_pred_proba = clf.predict_proba(X_test_validation)
    BA_lsvc_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_fullFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf, n_jobs = -1)
    forest.fit(X_train_FS, y_train_FS)
    y_pred = forest.predict(X_test_validation)
    y_pred_proba = forest.predict_proba(X_test_validation)
    BA_RF_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_fullFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0, n_jobs = -1)
    LogReg.fit(X_train_FS, y_train_FS)
    y_pred = LogReg.predict(X_test_validation)
    y_pred_proba = LogReg.predict_proba(X_test_validation)
    BA_LogReg_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_fullFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))
    
    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN)
    # kNN.fit(X_train_FS, y_train_FS)
    # y_pred = kNN.predict(X_test_validation)
    # y_pred_proba = kNN.predict_proba(X_test_validation)
    # BA_kNN_fullFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    # ROC_kNN_fullFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]])

    # Reduced feature set
    X_train_FS_reduced = X_train_FS[reduced_feature_names]
    X_test_validation_reduced = X_test_validation[reduced_feature_names]
    
    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    lsvc.fit(X_train_FS_reduced, y_train_FS)
    clf = CalibratedClassifierCV(lsvc) 
    clf.fit(X_train_FS_reduced, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_reduced)
    y_pred_proba = clf.predict_proba(X_test_validation_reduced)
    BA_lsvc_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_reducedFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf, n_jobs = -1)
    forest.fit(X_train_FS_reduced, y_train_FS)
    y_pred = forest.predict(X_test_validation_reduced)
    y_pred_proba = forest.predict_proba(X_test_validation_reduced)
    BA_RF_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_reducedFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0, n_jobs = -1)
    LogReg.fit(X_train_FS_reduced, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_reduced)
    y_pred_proba = LogReg.predict_proba(X_test_validation_reduced)
    BA_LogReg_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_reducedFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))
    
    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN)
    # kNN.fit(X_train_FS, y_train_FS)
    # y_pred = kNN.predict(X_test_validation_reduced)
    # y_pred_proba = kNN.predict_proba(X_test_validation_reduced)
    # BA_kNN_reducedFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    # ROC_kNN_reducedFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    # Reduced feature set permuted
    X_train_FS_reduced = X_train_FS[reduced_feature_names]
    X_test_validation_reduced = X_test_validation[reduced_feature_names]
    
    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    lsvc.fit(X_train_FS_reduced, y_train_FS)
    clf = CalibratedClassifierCV(lsvc) 
    clf.fit(X_train_FS_reduced, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_reduced)
    y_pred_proba = clf.predict_proba(X_test_validation_reduced)
    BA_lsvc_reducedFeatureSet_permuted.append(balanced_accuracy_score(y_test_validation_permuted, y_pred))
    ROC_lsvc_reducedFeatureSet_permuted.append(roc_auc_score(y_test_validation_num_permuted, y_pred_proba[:,[1]]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf, n_jobs = -1)
    forest.fit(X_train_FS_reduced, y_train_FS)
    y_pred = forest.predict(X_test_validation_reduced)
    y_pred_proba = forest.predict_proba(X_test_validation_reduced)
    BA_RF_reducedFeatureSet_permuted.append(balanced_accuracy_score(y_test_validation_permuted, y_pred))
    ROC_RF_reducedFeatureSet_permuted.append(roc_auc_score(y_test_validation_num_permuted, y_pred_proba[:,[1]]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0, n_jobs = -1)
    LogReg.fit(X_train_FS_reduced, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_reduced)
    y_pred_proba = LogReg.predict_proba(X_test_validation_reduced)
    BA_LogReg_reducedFeatureSet_permuted.append(balanced_accuracy_score(y_test_validation_permuted, y_pred))
    ROC_LogReg_reducedFeatureSet_permuted.append(roc_auc_score(y_test_validation_num_permuted, y_pred_proba[:,[1]]))
    
    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN)
    # kNN.fit(X_train_FS_reduced, y_train_FS)
    # y_pred = kNN.predict(X_test_validation_reduced)
    # y_pred_proba = kNN.predict_proba(X_test_validation_reduced)
    # BA_kNN_reducedFeatureSet_permuted.append(balanced_accuracy_score(y_test_validation_permuted, y_pred))
    # ROC_kNN_reducedFeatureSet_permuted.append(roc_auc_score(y_test_validation_num_permuted, y_pred_proba[:,[1]]))

    # Sparse feature set
    X_train_FS_sparse = X_train_FS[sparse_feature_names]
    X_test_validation_sparse = X_test_validation[sparse_feature_names]
    
    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C = C_svs, kernel = kernel_svs, gamma = gamma_svs, tol = tol_svs , max_iter=10000,random_state=0)
    lsvc.fit(X_train_FS_sparse, y_train_FS)
    clf = CalibratedClassifierCV(lsvc) 
    clf.fit(X_train_FS_sparse, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_sparse)
    y_pred_proba = clf.predict_proba(X_test_validation_sparse)
    BA_lsvc_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_sparseFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf, 
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_sparse, y_train_FS)
    y_pred = forest.predict(X_test_validation_sparse)
    y_pred_proba = forest.predict_proba(X_test_validation_sparse)
    BA_RF_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_sparseFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg, solver= solver_LogReg, tol=tol_LogReg, max_iter=10000,random_state=0)
    LogReg.fit(X_train_FS_sparse, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_sparse)
    y_pred_proba = LogReg.predict_proba(X_test_validation_sparse)
    BA_LogReg_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_sparseFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))

    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN, algorithm = algorithm_kNN, p = p_kNN, leaf_size= leaf_size_kNN, metric = metric_kNN )
    # kNN.fit(X_train_FS_sparse, y_train_FS)
    # y_pred = kNN.predict(X_test_validation_sparse)
    # y_pred_proba = kNN.predict_proba(X_test_validation_sparse)
    # BA_kNN_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    # ROC_kNN_sparseFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[:,[1]]))


CV_results_BA = pd.DataFrame(np.column_stack([BA_lsvc_fullFeatureSet, BA_RF_fullFeatureSet, BA_LogReg_fullFeatureSet, 
                                           BA_lsvc_reducedFeatureSet, BA_RF_reducedFeatureSet, BA_LogReg_reducedFeatureSet, 
                                           BA_lsvc_reducedFeatureSet_permuted, BA_RF_reducedFeatureSet_permuted, BA_LogReg_reducedFeatureSet_permuted, 
                                           BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet]),
                          columns = ["BA_lsvc_fullFeatureSet", "BA_RF_fullFeatureSet", "BA_LogReg_fullFeatureSet", 
                                                                     "BA_lsvc_reducedFeatureSet", "BA_RF_reducedFeatureSet", "BA_LogReg_reducedFeatureSet", 
                                                                     "BA_lsvc_reducedFeatureSet_permuted", "BA_RF_reducedFeatureSet_permuted", "BA_LogReg_reducedFeatureSet_permuted", 
                                                                     "BA_lsvc_sparseFeatureSet", "BA_RF_sparseFeatureSet", "BA_LogReg_sparseFeatureSet"])

CV_results_BA.mean()
CV_results_BA.std()
CV_results_BA.quantile()
CV_results_BA.quantile(0.025)
CV_results_BA.quantile(0.975)

fig, ax = plt.subplots(figsize=(18, 16))
sns.boxplot(ax = ax, data = CV_results_BA)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

CV_results_ROC = pd.DataFrame(np.column_stack([ROC_lsvc_fullFeatureSet, ROC_RF_fullFeatureSet, ROC_LogReg_fullFeatureSet, 
                                           ROC_lsvc_reducedFeatureSet, ROC_RF_reducedFeatureSet, ROC_LogReg_reducedFeatureSet, 
                                           ROC_lsvc_reducedFeatureSet_permuted, ROC_RF_reducedFeatureSet_permuted, ROC_LogReg_reducedFeatureSet_permuted, 
                                           ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet]),
                          columns = ["ROC_lsvc_fullFeatureSet", "ROC_RF_fullFeatureSet", "ROC_LogReg_fullFeatureSet", 
                                                                     "ROC_lsvc_reducedFeatureSet", "ROC_RF_reducedFeatureSet", "ROC_LogReg_reducedFeatureSet", 
                                                                     "ROC_lsvc_reducedFeatureSet_permuted", "ROC_RF_reducedFeatureSet_permuted", "ROC_LogReg_reducedFeatureSet_permuted", 
                                                                     "ROC_lsvc_sparseFeatureSet", "ROC_RF_sparseFeatureSet", "ROC_LogReg_sparseFeatureSet"])

CV_results_ROC.mean()
CV_results_ROC.std()
CV_results_ROC.quantile()
CV_results_ROC.quantile(0.025)
CV_results_ROC.quantile(0.975)

fig, ax = plt.subplots(figsize=(18, 16))
sns.boxplot(ax = ax, data = CV_results_ROC)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# %%



