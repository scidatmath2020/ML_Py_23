# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:52:02 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data")

mi_data_pixeles = pd.read_csv("mnist_pixeles.csv",header=None)
mi_data_clases = pd.read_csv("mnist_clases.csv",header=None)

#%%

mi_data_clases.shape
mi_data_pixeles.shape

#%%

primer_digito = mi_data_pixeles.iloc[0]
primer_digito.to_numpy()
plt.imshow(primer_digito.to_numpy().reshape(28,28), cmap="Greys")

mi_data_clases.iloc[0]

#%%
'''Analizar balanceo'''
mi_data_clases.value_counts()*100/mi_data_clases.shape[0]

#%%

from sklearn.decomposition import PCA
pca = PCA(0.8)

mnist_pca = pca.fit_transform(mi_data_pixeles)
mnist_pca.shape

#%%

from scipy.stats import randint as sp_randint

clf = KNeighborsClassifier()

busqueda_dist_parametros = {
    "n_neighbors": sp_randint(2,10),
    "p": sp_randint(1,3),
    "weights": ["uniform", "distance"]
}

from sklearn.model_selection import RandomizedSearchCV

busqueda = RandomizedSearchCV(estimator=clf,
                             param_distributions=busqueda_dist_parametros,
                             n_iter=3,
                             cv=3,
                             n_jobs=-1,
                             scoring="f1_micro")
busqueda.fit(X=mnist_pca, y=mi_data_clases.values.ravel())

busqueda.best_score_
busqueda.best_params_

#%%

mejores_params = {'n_neighbors': 7, 'p': 2, 'weights': 'distance'}

mejor_knn = KNeighborsClassifier(**mejores_params)
mejor_knn.fit(mnist_pca, mi_data_clases.values.ravel())


#%%

mi_numero = pd.read_csv("mi_numero.csv",header = None)
mi_numero.iloc[0].to_numpy()
plt.imshow(mi_numero.iloc[1].to_numpy().reshape(28,28), cmap="Greys")

nuevos_pca = pca.transform(mi_numero)
mejor_knn.predict(nuevos_pca)


#%% 
'''usar el modulo pickle para guardar y cargar'''

import pickle
with open("pca.pickle", "wb") as file:
    pickle.dump(pca, file)
    
with open("mejor_knn.pickle", "wb") as file:
    pickle.dump(mejor_knn, file)

#%%

import pickle
with open('pca.pickle', "rb") as file:
    mi_pca = pickle.load(file)
    
with open('mejor_knn.pickle', "rb") as file:
    mejor_knn = pickle.load(file)

#%%

nuevos_numeros = pd.read_csv("nuevos_numeros.csv",header = None)
nuevos_numeros_pca = mi_pca.transform(nuevos_numeros)
mejor_knn.predict(nuevos_numeros_pca)


plt.imshow(nuevos_numeros.iloc[0].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[1].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[2].to_numpy().reshape(28,28), cmap="Greys")
plt.imshow(nuevos_numeros.iloc[3].to_numpy().reshape(28,28), cmap="Greys")