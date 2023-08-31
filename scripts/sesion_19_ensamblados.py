# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:32:37 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from siuba import *

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC


os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("datos_peliculas.csv")
mi_data = mi_data >> mutate(objetivo = if_else(_.genero.isin([1,3,8]),_.genero,0))


#%%

variables_independientes = mi_data >> select(-_["pelicula","año","genero","objetivo"])
variable_objetivo = mi_data >> select(_.objetivo)

#%%


mi_data_normalizado = pd.DataFrame(StandardScaler().fit_transform(variables_independientes))

variable_objetivo.value_counts()*100/variable_objetivo.shape[0]


#%%

def evaluar_modelo(estimador, X, y):
    resultados_estimador = cross_validate(estimador, X, y,
                     scoring="f1_micro", n_jobs=-1, cv=30)
    return resultados_estimador

def ver_resultados():
    resultados_df  = pd.DataFrame(resultados).T
    resultados_cols = resultados_df.columns
    for col in resultados_df:
        resultados_df[col] = resultados_df[col].apply(np.mean)
        resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()
    return resultados_df >> arrange(-_.test_score,_.fit_time)

#%%

from sklearn.model_selection import RandomizedSearchCV

parametros_knn = {
    "n_neighbors": [1,10,20,30,40,50],
    "p": [1,2,3],
    "weights": ["uniform", "distance"]
}

parametros_svm_poly = {
    "degree": [1,2,3,4],
    "kernel": ["poly"]
}

parametros_svm_gauss = {
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["rbf"]
}

parametros_arbol = {
    "max_depth": list(range(3,6)),
    "criterion": ["gini","entropy"],    
    "class_weight": [None,"balanced"]
    }

#%%

estimador_knn = KNeighborsClassifier()
estimador_svm = SVC()
estimador_arbol = tree.DecisionTreeClassifier()
estimador_arbol_aleatorio = tree.ExtraTreeClassifier()

knn_grid = RandomizedSearchCV(estimator=estimador_knn, 
                    param_distributions=parametros_knn,
                    scoring="f1_micro", n_jobs=-1)

svm_poly_grid = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_poly,
                    scoring="f1_micro", n_jobs=-1)

svm_gauss_grid = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_svm_gauss,
                    scoring="f1_micro", n_jobs=-1)

arbol_grid = RandomizedSearchCV(estimator=estimador_arbol, 
                    param_distributions=parametros_arbol,
                    scoring="f1_micro", n_jobs=-1)
#%%

knn_grid.fit(variables_independientes, variable_objetivo.values.ravel())
svm_poly_grid.fit(variables_independientes, variable_objetivo.values.ravel())
svm_gauss_grid.fit(variables_independientes, variable_objetivo.values.ravel())
arbol_grid.fit(variables_independientes, variable_objetivo.values.ravel())

#%%

arbol_grid.best_estimator_

#%%

resultados = {}

resultados["knn"] = evaluar_modelo(knn_grid.best_estimator_,
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

resultados["svm_poly"] = evaluar_modelo(svm_poly_grid.best_estimator_,
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

resultados["svm_gauss"] = evaluar_modelo(svm_gauss_grid.best_estimator_,
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

resultados["arbol"] = evaluar_modelo(arbol_grid.best_estimator_,
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

resultados["arbol_aleatorio"] = evaluar_modelo(estimador_arbol_aleatorio,
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

pd.set_option('display.expand_frame_repr', False)
ver_resultados()

#%%
from sklearn.ensemble import BaggingClassifier

estimador_bagging_10 = BaggingClassifier(n_estimators=10)

score_bagging_10 = cross_validate(estimador_bagging_10,
                                  X=variables_independientes,
                                  y=variable_objetivo.values.ravel(), 
                                  scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_10

#%%

from sklearn.ensemble import BaggingClassifier

estimador_bagging_100 = BaggingClassifier(n_estimators=100)

score_bagging_100 = cross_validate(estimador_bagging_100,
                                   X=variables_independientes,
                                   y=variable_objetivo.values.ravel(), 
                                   scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_100

#%%

estimador_bagging_svm = BaggingClassifier(n_estimators = 100,
                                          base_estimator = SVC(kernel="rbf"))

score_bagging_svm = cross_validate(estimador_bagging_svm,
                                   X=variables_independientes,
                                   y=variable_objetivo.values.ravel(), 
                                   scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_svm

#%%

estimador_bagging_knn = BaggingClassifier(n_estimators=100,
                                          base_estimator=KNeighborsClassifier())
score_bagging_knn = cross_validate(estimador_bagging_knn,
                                   X=variables_independientes,
                                   y=variable_objetivo.values.ravel(), 
                                   scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_knn

#%%

estimador_bagging_mejor_arbol = BaggingClassifier(n_estimators=100,
                                          base_estimator=arbol_grid.best_estimator_)
score_bagging_mejor_arbol = cross_validate(estimador_bagging_mejor_arbol,
                                           X=variables_independientes,
                                           y=variable_objetivo.values.ravel(),
                                           scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_mejor_arbol

#%%

estimador_bagging_arbol_aleatorio = BaggingClassifier(n_estimators=500,
                                          base_estimator=tree.ExtraTreeClassifier())

score_bagging_arbol_aleatorio = cross_validate(estimador_bagging_arbol_aleatorio,
                                           X=variables_independientes,
                                           y=variable_objetivo.values.ravel(),
                                           scoring="f1_micro", cv=30)["test_score"].mean()

score_bagging_arbol_aleatorio

#%%

from sklearn.ensemble import AdaBoostClassifier

estimador_adaboost = AdaBoostClassifier(n_estimators=100)

score_adaboost = cross_validate(estimador_adaboost,
                                X=variables_independientes,
                                y=variable_objetivo.values.ravel(), 
                                scoring="f1_micro", cv=30)["test_score"].mean()

score_adaboost

#%%

from sklearn.ensemble import GradientBoostingClassifier

estimador_gradientboost = GradientBoostingClassifier(n_estimators=100, loss='log_loss')

score_gradientboost = cross_validate(estimador_gradientboost,
                                     X=variables_independientes,
                                     y=variable_objetivo.values.ravel(), 
                                     scoring="f1_micro", cv=30)["test_score"].mean()

score_gradientboost

#%%

from sklearn.ensemble import RandomForestClassifier

estimador_randomforest = RandomForestClassifier(n_estimators=100)

score_randomforest = cross_validate(estimador_randomforest,
                                    X=variables_independientes,
                                    y=variable_objetivo.values.ravel(), 
                                    scoring="f1_micro", cv=30)["test_score"].mean()

score_randomforest