# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:48:47 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################              Algoritmos knn              #####################
################                                          #####################
###############################################################################
'''


import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *


#%%

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("datos_peliculas.csv")

#%%

mi_data.head()
mi_data.columns
mi_data.shape

mi_data["año"].max()
mi_data["secuela"].unique()
mi_data["año"].unique()

peliculas = mi_data >> select(_.pelicula)
mi_data = mi_data >> select(-_.pelicula,-_.secuela)

#%%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#%%
'''
CLASIFICACIÓN

Probamos KNN para clasificación; en concreto vamos a suponer que queremos 
predecir el género de una película en función de las otras columnas
'''

sorted(mi_data["genero"].unique())

mi_data.shape

variable_objetivo_clasificacion = mi_data >> select(_.genero)
variables_independientes_clasificacion = mi_data >> select(-_.genero)





X_train, X_test, y_train, y_test = train_test_split(
    variables_independientes_clasificacion,
    variable_objetivo_clasificacion, test_size=0.20,random_state=2023)

sorted(y_train["genero"].unique())

'''Utilizando pesos uniformes'''
clasificador_knn_uniforme = KNeighborsClassifier(n_neighbors=3, weights="uniform")
clasificador_knn_uniforme.fit(X_train, y_train["genero"])

preds_uniforme = clasificador_knn_uniforme.predict(X_test)
f1_score(y_test, preds_uniforme, average="micro")

'''Utilizando pesos = "distancias" '''
clasificador_knn_distancias = KNeighborsClassifier(n_neighbors=10, weights="distance")
clasificador_knn_distancias.fit(X_train, y_train["genero"])

preds_distancias = clasificador_knn_distancias.predict(X_test)
f1_score(y_test, preds_distancias, average="micro")

#%%
'''Selección de k'''

def clasificadores_knn(k):
    knn_uniforme = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_distancias = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_uniforme.fit(X_train, y_train["genero"])
    knn_distancias.fit(X_train, y_train["genero"])
    preds_uniforme = knn_uniforme.predict(X_test)
    preds_distancias = knn_distancias.predict(X_test)
    f1_uniforme = f1_score(y_test, preds_uniforme, average="micro")
    f1_distancias = f1_score(y_test, preds_distancias, average="micro")
    return (k,f1_uniforme,f1_distancias)

k_limite = mi_data.shape[0] ** 0.5
k_limite

clasificacion_evaluaciones =[ clasificadores_knn(k) for k in range(1,int(k_limite),2)]

clasificacion_evaluaciones = pd.DataFrame(clasificacion_evaluaciones,
                                          columns = ["k","F1_uniforme","F1_distancias"])
#%%
clasificaciones_evaluaciones_tidy = clasificacion_evaluaciones >> gather("F1_tipo",
                                                                         "F1_score",
                                                                         -_.k)

(ggplot(data = clasificaciones_evaluaciones_tidy) +
    geom_point(mapping=aes(x="k",y="F1_score",color="F1_tipo")) +
    geom_line(mapping=aes(x="k",y="F1_score",color="F1_tipo"))
)


(ggplot(data = clasificacion_evaluaciones) +
    geom_point(mapping=aes(x="k",y="F1_uniforme"),color = "red") +
    geom_line(mapping=aes(x="k",y="F1_uniforme"),color = "red") +
    geom_point(mapping=aes(x="k",y="F1_distancias"),color = "blue") +
    geom_line(mapping=aes(x="k",y="F1_distancias"),color = "blue")
)


(clasificacion_evaluaciones >> 
    filter((_.F1_uniforme == _.F1_uniforme.max()) | (_.F1_distancias == _.F1_distancias.max()))
)

#%%

'''Utilizando pesos uniformes'''
mejor_clasificador_knn_uniforme = KNeighborsClassifier(n_neighbors=15, weights="uniform")
mejor_clasificador_knn_uniforme.fit(X_train, y_train["genero"])

mejor_preds_uniforme = mejor_clasificador_knn_uniforme.predict(X_test)
f1_score(y_test, mejor_preds_uniforme, average="micro")

'''Utilizando pesos = "distancias" '''
mejor_clasificador_knn_distancias = KNeighborsClassifier(n_neighbors=17, weights="distance")
mejor_clasificador_knn_distancias.fit(X_train, y_train["genero"])

mejor_preds_distancias = mejor_clasificador_knn_distancias.predict(X_test)
f1_score(y_test, mejor_preds_distancias, average="micro")

#%%

nvo_dato = {"año":[2014],
            "ratings":[876],
            "ventas":[87876876],
            "presupuesto":[823223],
            "vistas_youtube":[76786876],
            'positivos_youtube':[76767], 
            'negativos_youtube':[9094], 
            'comentarios':[8878],
            'seguidores_agregados':[6786548]
            }


nvo_dato = pd.DataFrame(nvo_dato)

mejor_clasificador_knn_distancias.predict(nvo_dato)



