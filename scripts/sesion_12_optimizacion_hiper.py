# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:38:33 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("datos_iris.csv")

#%%

variables_independientes = mi_data >> select(-_.Species)
variable_objetivo = mi_data >> select(_.Species)

#%%

def evaluar_modelo(estimador, X, y):
    resultados_estimador = cross_validate(estimador, X, y,
                     scoring="f1_micro", n_jobs=-1, cv=5)
    return resultados_estimador

def ver_resultados():
    resultados_df  = pd.DataFrame(resultados).T
    resultados_cols = resultados_df.columns
    for col in resultados_df:
        resultados_df[col] = resultados_df[col].apply(np.mean)
        resultados_df[col+"_idx"] = resultados_df[col] / resultados_df[col].max()
    return resultados_df >> arrange(-_.test_score,_.fit_time)

#%%

resultados = {}

resultados["knn"] = evaluar_modelo(KNeighborsClassifier(),
                                   variables_independientes,
                                   variable_objetivo.values.ravel())
resultados["arbol_clasificacion"] = evaluar_modelo(tree.DecisionTreeClassifier(),
                                   variables_independientes,
                                   variable_objetivo.values.ravel())
resultados["arbol_aleatorio"] = evaluar_modelo(tree.ExtraTreeClassifier(),
                                   variables_independientes,
                                   variable_objetivo.values.ravel())
resultados["msv"] = evaluar_modelo(SVC(),
                                   variables_independientes,
                                   variable_objetivo.values.ravel())

pd.set_option('display.expand_frame_repr', False)
ver_resultados()

#%%

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#%%
estimador_knn = KNeighborsClassifier()
print(estimador_knn.__doc__)
estimador_knn.get_params()

#%%
parametros_busqueda_knn = {
    "n_neighbors": [1,10,20,30,40,50],
    "p": [1,2,3],
    "weights": ["uniform", "distance"]
}


knn_grid = GridSearchCV(estimator=estimador_knn, 
                    param_grid=parametros_busqueda_knn,
                    scoring="f1_micro", n_jobs=-1)

start_time = time.time()

knn_grid.fit(variables_independientes, variable_objetivo.values.ravel())

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

print(knn_grid.best_score_)
print(knn_grid.best_estimator_.get_params())

resultados["knn_gridsearch"] = evaluar_modelo(knn_grid.best_estimator_,
                                             variables_independientes,
                                             variable_objetivo.values.ravel())

ver_resultados()

#%%
knn_random = RandomizedSearchCV(estimator=estimador_knn, 
                    param_distributions=parametros_busqueda_knn,
                   scoring="f1_micro", n_jobs=-1, n_iter=10)

start_time = time.time()

knn_random.fit(variables_independientes, variable_objetivo.values.ravel())

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

print(knn_random.best_score_)
print(knn_random.best_estimator_)

resultados["knn_randomizedsearch"] = evaluar_modelo(knn_random.best_estimator_,
                                                    variables_independientes,
                                                    variable_objetivo.values.ravel())

ver_resultados()

#%%

estimador_svm = SVC()
print(estimador_svm.__doc__)
estimador_svm.get_params()

parametros_busqueda_svm = {
    "degree": [1,2,3,4],
    "gamma": [0.1,0.5,1.,10.],
    "kernel": ["poly", "rbf"]
}


#%%
svm_grid = GridSearchCV(estimator=estimador_svm, 
                    param_grid=parametros_busqueda_svm,
                    scoring="f1_micro", n_jobs=-1)
svm_grid.fit(variables_independientes, variable_objetivo.values.ravel())

print(svm_grid.best_estimator_)
resultados["svm_gridsearch"] = evaluar_modelo(svm_grid.best_estimator_,
                                             variables_independientes,
                                             variable_objetivo.values.ravel())

ver_resultados()

#%%
svm_random = RandomizedSearchCV(estimator=estimador_svm, 
                    param_distributions=parametros_busqueda_svm,
                   scoring="f1_micro", n_jobs=-1, n_iter=10)

svm_random.fit(variables_independientes, variable_objetivo.values.ravel())

print(svm_random.best_estimator_)
resultados["svm_randomizedsearch"] = evaluar_modelo(svm_random.best_estimator_,
                                             variables_independientes,
                                             variable_objetivo.values.ravel())

ver_resultados()

#%%

ver_resultados() 

