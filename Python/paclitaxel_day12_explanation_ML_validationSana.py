#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 16:14:53 2022

@author: joern
"""
# https://coderzcolumn.com/tutorials/machine-learning/scikit-learn-sklearn-feature-selection

# %% imports
import os
os.chdir("/home/joern/Aktuell/PaclitaxelPainLipidomics/08AnalyseProgramme/PaclitaxelNeuropathyProject/Python/")

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
from cABCanalysis import  cABCanalysis
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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

# %% Read scaled data and classes from R
pfad_o1 = "/home/joern/Aktuell/PaclitaxelPainLipidomics/"
pfad_u2 = "08AnalyseProgramme/Python/"
pfad_u3 = "08AnalyseProgramme/R/"

dfPaclitaxel = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_scaled.csv")
dfPaclitaxel.columns
dfPaclitaxel.set_index("Unnamed: 0", inplace=True)
dfPaclitaxel.columns
len(dfPaclitaxel.columns)

dfPaclitaxel.index

Paclitaxel_targets = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_Classes.csv")
Paclitaxel_targets .columns

# Sana data 

dfPaclitaxel_sana = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_sana_scaled.csv")
dfPaclitaxel_sana.columns
dfPaclitaxel_sana.set_index("Unnamed: 0", inplace=True)
dfPaclitaxel_sana.columns
len(dfPaclitaxel_sana.columns)

dfPaclitaxel_sana.index

Paclitaxel_targets_sana = pd.read_csv(pfad_o1 + pfad_u3 + "dfLipids_Classes_sana.csv")
Paclitaxel_targets_sana .columns


# %% Cluster explnation ML

y = Paclitaxel_targets["Probe12"]
pd.DataFrame(y).value_counts()


X_train, X_test, y_train, y_test = train_test_split(
    dfPaclitaxel, y, test_size=0.2, random_state=42)
pd.DataFrame(y_test).value_counts()
# https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6

y_sana = Paclitaxel_targets_sana["Probe12"]
pd.DataFrame(y_sana).value_counts()


X_train, X_test, y_train, y_test = train_test_split(
    dfPaclitaxel, y, test_size=0.2, random_state=42)
pd.DataFrame(y_test).value_counts()



# %% Select only lipids that are in both uct and sana samples'

LipidVariableNames_sana = pd.read_csv(pfad_o1 + pfad_u3 + "LipidVariableNames_sana.csv")
LipidVariableNames_sana_l = LipidVariableNames_sana["x"].to_list()
len(LipidVariableNames_sana_l)

X_train_l = X_train[LipidVariableNames_sana_l]
X_train_l.columns
X_test_l = X_train[LipidVariableNames_sana_l]
X_test_l.columns

print(X_train.shape)
print(X_train_l.shape)
print(X_test.shape)
print(X_test_l.shape)



# %% Create results table all methods and variables
featureSelection_methods = ["PCA", "features_CohenD", "features_FPR", "features_FWE",
                            "features_lSVC_sKb", "features_RF_sKb", "features_LogReg_sKb",
                            "features_lSVC_sfm", "features_RF_sfm", "features_LogReg_sfm",
                            "features_lSVC_rfe", "features_RF_rfe", "features_LogReg_rfe",
                            "features_lSVC_sfs_forward", "features_RF_sfs_forward", "features_LogReg_sfs_forward",
                            "features_lSVC_sfs_backward", "features_RF_sfs_backward", "features_LogReg_sfs_backward"]

feature_table = pd.DataFrame(
    np.zeros((len(X_train.columns), len(featureSelection_methods))))
feature_table.columns = featureSelection_methods
feature_table.set_index(X_train.columns, inplace=True)


# %% CV
n_splits = 5
n_repeats = 20


# %% Reload the results from file to run validation without repeating feature selection
feature_table_day12a = pd.read_csv(
    "/home/joern/Aktuell/PaclitaxelPainLipidomics/08AnalyseProgramme/Python/feature_table_day12.csv")
feature_table_day12a.columns

feature_table_day12a.set_index("Unnamed: 0", inplace=True)
feature_table_day12a.index

# %% feature selection sum score for ABC for selection of final feature set

FS_sumscore_day12 = feature_table_day12a.sum(axis=1)
FS_sumscore_day12.sort_values(ascending=False, inplace=True)

with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(12, 14))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, wspace=.1, hspace=0.4)

    ax2 = fig.add_subplot(gs0[0, 0])
    ax1 = fig.add_subplot(gs0[0, 1])
    axes = [ax2, ax1]
    for i, ax in enumerate(axes):
        annotate_axes(ax,  str(string.ascii_lowercase[i]) + ")")

    ABC_A_FS_sumscore =  cABCanalysis(
        ax=ax1, data=FS_sumscore_day12, PlotIt=True)
    ABC_A_FS_sumscore_nested =  cABCanalysis(ABC_A_FS_sumscore["Aind"]["value"])

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

ABC_A_FS_sumscore =  cABCanalysis(
    ax=ax1, data=FS_sumscore_day12, PlotIt=True)
ABC_A_FS_sumscore_nested =  cABCanalysis(
    ax=ax3, data=ABC_A_FS_sumscore["Aind"]["value"], PlotIt=True)

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

# file = "reduced_feature_names_day12.csv"
# ABC_A_FS_sumscore["Aind"].to_csv(file)
# file = "sparse_feature_names_day12.csv"
# ABC_A_FS_sumscore_nested["Aind"].to_csv(file

# %% Select only features that are available in sana data

sparse_feature_names_l =  intersection(sparse_feature_names, LipidVariableNames_sana_l)

print(len(sparse_feature_names))
print(len(sparse_feature_names_l))


# %% Classifier tuning

X_train_l_sparse = X_train_l[sparse_feature_names_l]

# LinearSVC
lsvc = LinearSVC(max_iter=10000, random_state=0)
param_grid = {"C": np.arange(0.01, 100, 10),
              "penalty": ["l1", "l2"],
              "dual": [True, False],
              "loss": ["hinge", "squared_hinge"],
              "tol": [0.001, 0.0001, 0.00001]}
grid_search = GridSearchCV(lsvc, param_grid=param_grid,
                           scoring="balanced_accuracy", verbose=0, n_jobs=-1)
grid_search.fit(X_train_l_sparse, y_train)
C_lsvm, dual_svm, loss_svm, penalty_SVM, tol_svm = grid_search.best_params_.values()

# SVM SVC
svc = SVC(max_iter=10000, random_state=0)
param_grid = {"C": np.arange(0.01, 100, 10),
              "kernel": ["linear", "poly", "rbf", "sigmoid"],
              "gamma": [1e-3, 1e-4, "scale", "auto"],
              "tol": [0.001, 0.0001, 0.00001]}
grid_search = GridSearchCV(svc, param_grid=param_grid,
                           scoring="balanced_accuracy", verbose=0, n_jobs=-1)
grid_search.fit(X_train_l_sparse, y_train)
C_svs, gamma_svs, kernel_svs, tol_svs = grid_search.best_params_.values()

# Random forests
forest = RandomForestClassifier(random_state=0)
param_grid = {'bootstrap': [True, False],
              'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1, 2, 4],
              'min_samples_split': [2, 5, 10],
              'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
grid_search = GridSearchCV(forest, param_grid=param_grid,
                           scoring="balanced_accuracy", verbose=0, n_jobs=-1)
grid_search.fit(X_train_l_sparse, y_train)
bootstrap_rf,  max_depth_rf, max_features_rf, min_samples_leaf_rf, min_samples_split_rf, n_estimators_rf = grid_search.best_params_.values()

# Logistic regregssion
LogReg = LogisticRegression(max_iter=10000, random_state=0)
param_grid = {"C": np.logspace(-3, 3, 7),
              "tol": [0.001, 0.0001, 0.00001],
              "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
grid_search = GridSearchCV(LogReg, param_grid=param_grid,
                           scoring="balanced_accuracy", verbose=0, n_jobs=-1)
grid_search.fit(X_train_l_sparse, y_train)
C_LogReg,  solver_LogReg, tol_LogReg = grid_search.best_params_.values()
penalty_LogReg = "l2"

# kNN
kNN = KNeighborsClassifier()
param_grid = {"n_neighbors": [3, 5, 7, 9],
              "p": [1, 2, 3],
              "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
              "leaf_size": [10, 20, 30, 40, 50],
              "metric": ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", "mahalanobi"]}
grid_search = GridSearchCV(kNN, param_grid=param_grid,
                           scoring="balanced_accuracy", verbose=0, n_jobs=-1)
grid_search.fit(X_train_l_sparse, y_train)
algorithm_kNN, leaf_size_kNN, metric_kNN, n_neighbors_kNN, p_kNN, = grid_search.best_params_.values()



# %% Test of selected features whether they suffice to predict the separate validation data subset
# Balanced accuracy and ROC AUC

BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet, BA_kNN_sparseFeatureSet = [], [], [], []
BA_lsvc_sparseFeatureSet_permuted, BA_RF_sparseFeatureSet_permuted, BA_LogReg_sparseFeatureSet_permuted, BA_kNN_sparseFeatureSet_permuted = [], [], [], []
ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet, ROC_kNN_sparseFeatureSet = [], [], [], []
ROC_lsvc_sparseFeatureSet_permuted, ROC_RF_sparseFeatureSet_permuted, ROC_LogReg_sparseFeatureSet_permuted, ROC_kNN_sparseFeatureSet_permuted = [], [], [], []

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
for train_index, test_index in rskf.split(X_train, y_train):
    X_train_FS, X_test_FS = X_train.iloc[train_index,
                                         :], X_train.iloc[test_index, :]
    y_train_FS, y_test_FS = y_train.iloc[train_index], y_train.iloc[test_index]

    X_train_validation, X_test_validation, y_train_validation, y_test_validation = train_test_split(
        dfPaclitaxel_sana, y_sana, test_size=0.8)
    y_test_validation_num = [0 if x == 1 else 1 for x in y_test_validation]

    y_test_validation_permuted = y_test_validation.copy()
    y_test_validation_permuted = pd.Series(
        np.random.permutation(y_test_validation_permuted))
    y_test_validation_num_permuted = y_test_validation_num.copy()
    y_test_validation_num_permuted = pd.Series(
        np.random.permutation(y_test_validation_num_permuted))


    # Sparse feature set
    X_train_FS_sparse = X_train_FS[sparse_feature_names_l]
    X_test_validation_sparse = X_test_validation[sparse_feature_names_l]

    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C=C_svs, kernel=kernel_svs, gamma=gamma_svs,
               tol=tol_svs, max_iter=10000, random_state=0)
    lsvc.fit(X_train_FS_sparse, y_train_FS)
    clf = CalibratedClassifierCV(lsvc)
    clf.fit(X_train_FS_sparse, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_sparse)
    y_pred_proba = clf.predict_proba(X_test_validation_sparse)
    BA_lsvc_sparseFeatureSet.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_sparseFeatureSet.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf,
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_sparse, y_train_FS)
    y_pred = forest.predict(X_test_validation_sparse)
    y_pred_proba = forest.predict_proba(X_test_validation_sparse)
    BA_RF_sparseFeatureSet.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_sparseFeatureSet.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg,
                                solver=solver_LogReg, tol=tol_LogReg, max_iter=10000, random_state=0)
    LogReg.fit(X_train_FS_sparse, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_sparse)
    y_pred_proba = LogReg.predict_proba(X_test_validation_sparse)
    BA_LogReg_sparseFeatureSet.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_sparseFeatureSet.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN, algorithm = algorithm_kNN, p = p_kNN, leaf_size= leaf_size_kNN, metric = metric_kNN )
    # kNN.fit(X_train_FS_sparse, y_train_FS)
    # y_pred = kNN.predict(X_test_validation_sparse)
    # y_pred_proba = kNN.predict_proba(X_test_validation_sparse)
    # BA_kNN_sparseFeatureSet.append(balanced_accuracy_score(y_test_validation, y_pred))
    # ROC_kNN_sparseFeatureSet.append(roc_auc_score(y_test_validation_num, y_pred_proba[::,1]))


    # Sparse feature set permuted
    X_train_FS_sparse = X_train_FS[sparse_feature_names_l]
    X_test_validation_sparse = X_test_validation[sparse_feature_names_l]
    
    X_train_FS_sparse_permuted  = X_train_FS_sparse.copy()
    X_train_FS_sparse_permuted = X_train_FS_sparse_permuted.apply(lambda x: x.sample(frac=1).values)


    #lsvc = LinearSVC(C = C_lsvm, penalty = penalty_SVM, dual = dual_svm, loss = loss_svm, tol = tol_svm, max_iter=10000)
    lsvc = SVC(C=C_svs, kernel=kernel_svs, gamma=gamma_svs,
               tol=tol_svs, max_iter=10000, random_state=0)
    lsvc.fit(X_train_FS_sparse_permuted, y_train_FS)
    clf = CalibratedClassifierCV(lsvc)
    clf.fit(X_train_FS_sparse_permuted, y_train_FS)
    y_pred = lsvc.predict(X_test_validation_sparse)
    y_pred_proba = clf.predict_proba(X_test_validation_sparse)
    BA_lsvc_sparseFeatureSet_permuted.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_lsvc_sparseFeatureSet_permuted.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    forest = RandomForestClassifier(random_state=0, bootstrap=bootstrap_rf,  max_depth=max_depth_rf, max_features=max_features_rf,
                                    min_samples_leaf=min_samples_leaf_rf, min_samples_split=min_samples_split_rf, n_estimators=n_estimators_rf)
    forest.fit(X_train_FS_sparse_permuted, y_train_FS)
    y_pred = forest.predict(X_test_validation_sparse)
    y_pred_proba = forest.predict_proba(X_test_validation_sparse)
    BA_RF_sparseFeatureSet_permuted.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_RF_sparseFeatureSet_permuted.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    LogReg = LogisticRegression(C=C_LogReg,  penalty=penalty_LogReg,
                                solver=solver_LogReg, tol=tol_LogReg, max_iter=10000, random_state=0)
    LogReg.fit(X_train_FS_sparse_permuted, y_train_FS)
    y_pred = LogReg.predict(X_test_validation_sparse)
    y_pred_proba = LogReg.predict_proba(X_test_validation_sparse)
    BA_LogReg_sparseFeatureSet_permuted.append(
        balanced_accuracy_score(y_test_validation, y_pred))
    ROC_LogReg_sparseFeatureSet_permuted.append(roc_auc_score(
        y_test_validation_num, y_pred_proba[::, 1]))

    # kNN = KNeighborsClassifier(n_neighbors = n_neighbors_kNN, algorithm = algorithm_kNN, p = p_kNN, leaf_size= leaf_size_kNN, metric = metric_kNN )
    # kNN.fit(X_train_FS_sparse_permuted, y_train_FS)
    # y_pred = kNN.predict(X_test_validation_sparse)
    # y_pred_proba = kNN.predict_proba(X_test_validation_sparse)
    # BA_kNN_sparseFeatureSet_permuted.append(balanced_accuracy_score(y_test_validation, y_pred))
    # ROC_kNN_sparseFeatureSet_permuted.append(roc_auc_score(y_test_validation_num, y_pred_proba[::,1]))


CV_results_BA = pd.DataFrame(np.column_stack([BA_lsvc_sparseFeatureSet, BA_RF_sparseFeatureSet, BA_LogReg_sparseFeatureSet, 
                                              BA_lsvc_sparseFeatureSet_permuted, BA_RF_sparseFeatureSet_permuted, BA_LogReg_sparseFeatureSet_permuted]),
                             columns=["BA_lsvc_sparseFeatureSet", "BA_RF_sparseFeatureSet", "BA_LogReg_sparseFeatureSet",
                                      "BA_lsvc_sparseFeatureSet_permuted", "BA_RF_sparseFeatureSet_permuted", "BA_LogReg_sparseFeatureSet_permuted"])

CV_results_BA.mean()
CV_results_BA.std()
CV_results_BA.quantile()
CV_results_BA.quantile(0.025)
CV_results_BA.quantile(0.975)
CV_results_BA.to_csv(pfad_o1 + pfad_u2 +
                      "CV_results_BA_Paclitaxel_day12_validationSana.csv")
fig, ax = plt.subplots(figsize=(18, 16))
sns.boxplot(ax=ax, data=CV_results_BA)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

CV_results_ROC = pd.DataFrame(np.column_stack([ROC_lsvc_sparseFeatureSet, ROC_RF_sparseFeatureSet, ROC_LogReg_sparseFeatureSet,
                                               ROC_lsvc_sparseFeatureSet_permuted, ROC_RF_sparseFeatureSet_permuted, ROC_LogReg_sparseFeatureSet_permuted]),
                              columns=["ROC_lsvc_sparseFeatureSet", "ROC_RF_sparseFeatureSet", "ROC_LogReg_sparseFeatureSet",
                                       "ROC_lsvc_sparseFeatureSet_permuted", "ROC_RF_sparseFeatureSet_permuted", "ROC_LogReg_sparseFeatureSet_permuted"])

CV_results_ROC.mean()
CV_results_ROC.std()
CV_results_ROC.quantile()
CV_results_ROC.quantile(0.025)
CV_results_ROC.quantile(0.975)
CV_results_ROC.to_csv(pfad_o1 + pfad_u2 +
                      "CV_results_ROC_Paclitaxel_day12_validationSana.csv")


fig, ax = plt.subplots(figsize=(18, 16))
sns.boxplot(ax=ax, data=CV_results_ROC)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



# %% End

duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
os.system('play -nq -t alsa synth {} sine {}'.format(duration*.1, freq*2))
os.system('play -nq -t alsa synth {} sine {}'.format(duration*.1, freq*2))
os.system('play -nq -t alsa synth {} sine {}'.format(duration*.1, freq*2))