# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:15:29 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from siuba import *


#%%
ruta = "https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/datos_regresion.csv"
mi_tabla = pd.read_csv(ruta)

mi_tabla

(ggplot(data = mi_tabla) + 
 geom_point(mapping=aes(x="caracteristica_1",y="valor_real"),color="red")
)

#%%

variables_independientes = mi_tabla >> select(_.caracteristica_1)
objetivo = mi_tabla >> select(_.valor_real)

#%%

from sklearn.linear_model import LinearRegression
modelo = LinearRegression()

#%%

modelo.fit(X=variables_independientes,y=objetivo) # Aquí se calcula la regresión
modelo.intercept_ #esta es la alpha de la regresión
modelo.coef_ # estas son las beta's de la regresión 

#%%

mi_tabla["predicciones"] = modelo.predict(variables_independientes)

#%%

(ggplot(data = mi_tabla) +
 geom_point(mapping=aes(x="caracteristica_1",y="valor_real"),color="blue") +
 geom_point(mapping=aes(x="caracteristica_1",y="predicciones"),color="red") +
 geom_abline(slope=1.85,intercept=5.711) +
 geom_smooth(mapping=aes(x="caracteristica_1",y="valor_real"),color="green")
)

modelo.coef_
modelo.intercept_

#%%

from sklearn import metrics

metrics.mean_absolute_error(mi_tabla["valor_real"],mi_tabla["predicciones"])
metrics.mean_squared_error(mi_tabla["valor_real"],mi_tabla["predicciones"])
np.sqrt(metrics.mean_squared_error(mi_tabla["valor_real"],mi_tabla["predicciones"]))

mi_tabla >> mutate(error = _.valor_real-_.predicciones)

R2 = metrics.r2_score(mi_tabla["valor_real"],mi_tabla["predicciones"])

1-(1-R2)*(50-1)/(50-1-1)

#%%

n = mi_tabla.shape[0]
k = variables_independientes.shape[1]

1-(1-R2)*(n-1)/(n-k-1)

#%%

'''
###############################################################################
################                                          #####################
################         Regresión lineal completa        #####################
################                                          #####################
###############################################################################
'''

import pandas as pd
import numpy as np
from plotnine import *
from siuba import *
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#%%

import os
ruta = "C:\\Users\\hp master\\OneDrive\\Escritorio\\23_ml_py"
os.chdir(ruta)
mi_tabla = pd.read_csv("casas_boston.csv")


#%%

ruta = "https://raw.githubusercontent.com/scidatmath2020/ML_Py_23/main/data/casas_boston.csv"
mi_tabla_2 = pd.read_csv(ruta)

#%%

variables_independientes = mi_tabla >> select(-_.MEDV)
objetivo = mi_tabla >> select(_.MEDV)

#%%

modelo_regresion = LinearRegression()
modelo_regresion.fit(X=variables_independientes,y=objetivo)
mi_tabla = mi_tabla >> mutate(predicciones = modelo_regresion.predict(variables_independientes))

'''
MUY IMPORTANTE
En este momento mi_tabla es la tabla original PERO CON UNA COLUMNA EXTRA: LA DE LAS PREDICCIONES DE LA COMPUTADORA
'''


#%%

'''Función para evaluar el modelo. Sus argumentos son:
    - independientes: tabla de columnas predictoras (es la tabla azul)
    - nombre_columna_objetivo: es el nombre de la columna objetivo de la tabla original
    - tabla_full: es la tabla completa del comentario anterior'''

def evaluar_regresion(independientes,nco,tabla_full):
    n = independientes.shape[0]
    k = independientes.shape[1]
    mae = metrics.mean_absolute_error(tabla_full[nco],tabla_full["predicciones"])
    rmse = np.sqrt(metrics.mean_squared_error(tabla_full[nco],tabla_full["predicciones"]))
    r2 = metrics.r2_score(tabla_full[nco],tabla_full["predicciones"])
    r2_adj = 1-(1-r2)*(n-1)/(n-k-1)
    return {"r2_adj":r2_adj,"mae":mae,"rmse":rmse}
    
#%%
    
evaluar_regresion(variables_independientes,"MEDV",mi_tabla)

#%%

'''
###############################################################################
################                                          #####################
################ SEPARACION EN ENTRENAMIENTO Y PRUEBA     #####################
################                                          #####################
###############################################################################
'''

from sklearn.model_selection import train_test_split

#%%

'''Dividiemos en entrenamiento y prueba. El 33% de los datos es para prueba y
utilizamos una semilla igual a 13'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=13)

#%%
indepen_entrenamiento.shape[0]
objetivo_entrenamiento.shape[0]
indepen_prueba.shape[0]
objetivo_prueba.shape[0]

#%%

mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

#%%

modelo_entrenamiento = LinearRegression()

modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)

mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))

#%%
mi_tabla_entrenamiento.columns

evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)

#%%

mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))
evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)

#%%

Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)

Resultados = pd.DataFrame(Resultados)
Resultados

#%%
'''Cambiando random_state a 42'''

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                                objetivo,
                                                                                                                test_size=0.33,
                                                                                                                random_state=42)

mi_tabla_entrenamiento = indepen_entrenamiento >> mutate(objetivo = objetivo_entrenamiento)
mi_tabla_prueba = indepen_prueba >> mutate(objetivo = objetivo_prueba)

modelo_entrenamiento = LinearRegression()
modelo_entrenamiento.fit(X=indepen_entrenamiento,y=objetivo_entrenamiento)
mi_tabla_entrenamiento = mi_tabla_entrenamiento >> mutate(predicciones = modelo_entrenamiento.predict(indepen_entrenamiento))
mi_tabla_prueba = mi_tabla_prueba >> mutate(predicciones = modelo_entrenamiento.predict(indepen_prueba))

Resultados = {}
Resultados["tabla_original"] = evaluar_regresion(variables_independientes,"MEDV",mi_tabla)
Resultados["tabla_prueba"] = evaluar_regresion(indepen_prueba,"objetivo",mi_tabla_prueba)
Resultados["tabla_entrenamiento"] = evaluar_regresion(indepen_entrenamiento,"objetivo",mi_tabla_entrenamiento)


Resultados = pd.DataFrame(Resultados)
Resultados


#%%
'''
###############################################################################
################                                          #####################
#########################   Validación cruzada      ###########################
################                                          #####################
###############################################################################
'''

from sklearn.model_selection import cross_val_score

#%%

modelo_regresion_validacion = LinearRegression()

variables_independientes = mi_tabla >> select(-_.MEDV)
objetivo = mi_tabla >> select(_.MEDV)

cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=10)

rmse_validacion = [-cross_val_score(modelo_regresion_validacion,
                variables_independientes,
                objetivo,
                scoring = "neg_root_mean_squared_error",
                cv=x).mean() for x in range(10,150) 
]


evaluacion_cruzada = {"particiones":list(range(10,150)),
                      "rmse_validacion":rmse_validacion}

evaluacion_cruzada = pd.DataFrame(evaluacion_cruzada)

(ggplot(data = evaluacion_cruzada) +
 geom_line(mapping=aes(x="particiones",y="rmse_validacion")) 
 )
