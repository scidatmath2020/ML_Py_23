# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:13:38 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *

#%%

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("cancer_mama.csv")

#%%

mi_data.shape

mi_data.columns

#%%

mi_data >> group_by(_.diagnosis) >> summarize(conteo_objetivo = n(_), porcentaje_objetivo = n(_)/569)

#%%

mi_data = mi_data >> mutate(diagnosis = _.diagnosis.replace({0:1,1:0}))

#%%

from sklearn.linear_model import LinearRegression

data_peor_area = mi_data >> select(_.worst_area,_.diagnosis)

(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red")
 )

modelo_reg_lineal = LinearRegression()
modelo_reg_lineal.fit(X=data_peor_area >> select(_.worst_area),
                      y=data_peor_area >> select(_.diagnosis))


data_peor_area = data_peor_area >> mutate(predicciones_reg_lineal = modelo_reg_lineal.predict(data_peor_area >> select(_.worst_area)))

(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red") +
    geom_line(mapping = aes(x="worst_area", y="predicciones_reg_lineal"),color="blue")
 )

#%%

from sklearn.linear_model import LogisticRegression

modelo_reg_logis = LogisticRegression() 

modelo_reg_logis.fit(X=data_peor_area >> select(_.worst_area), y=data_peor_area["diagnosis"])


data_peor_area = data_peor_area >> mutate(probabilidades_reg_logis = (modelo_reg_logis.predict_proba(data_peor_area >> select(_.worst_area)))[:,1])
                         

(ggplot(data = data_peor_area) +
    geom_point(mapping = aes(x="worst_area", y="diagnosis"),color="red") +
    geom_line(mapping = aes(x="worst_area", y="predicciones_reg_lineal"),color="blue") +
    geom_line(mapping = aes(x="worst_area", y="probabilidades_reg_logis"),color="darkgreen")
 )

data_peor_area = data_peor_area >> mutate(prediccion = modelo_reg_logis.predict(data_peor_area >> select(_.worst_area)))


data_peor_area
#%%

'''
###############################################################################
################                                          #####################
################           Regreseión logística           #####################
################                                          #####################
###############################################################################
'''


import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("cancer_mama.csv")

mi_data = mi_data >> mutate(diagnosis = _.diagnosis.replace({0:1,1:0}))

#%%

'''separación entrenamiento y prueba'''

variables_independientes = mi_data >> select(-_.diagnosis)
objetivo = mi_data >> select(_.diagnosis)

indepen_entrenamiento, indepen_prueba, objetivo_entrenamiento, objetivo_prueba = train_test_split(variables_independientes,
                                                                                                  objetivo,
                                                                                                  test_size=0.3,
                                                                                                  random_state=42)


modelo_rl = LogisticRegression(solver = "newton-cholesky")
modelo_rl.fit(indepen_entrenamiento,objetivo_entrenamiento.values.ravel())

predicciones = modelo_rl.predict(indepen_prueba)
predicciones_probabilidades = modelo_rl.predict_proba(indepen_prueba)
objetivos_reales = objetivo_prueba.values.ravel()

#%%

def tupla_clase_prediccion(y_real, y_pred):
    return list(zip(y_real, y_pred))

tupla_clase_prediccion(objetivos_reales, predicciones)[:20]


#%%
def VP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==1])

def VN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==0])
    
def FP(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==0 and obs[1]==1])

def FN(clases_reales, predicciones):
    par_clase_prediccion = tupla_clase_prediccion(clases_reales, predicciones)
    return len([obs for obs in par_clase_prediccion if obs[0]==1 and obs[1]==0])


print("""
Verdaderos Positivos: {}
Verdaderos Negativos: {}
Falsos Positivos: {}
Falsos Negativos: {}
""".format(
    VP(objetivos_reales, predicciones),
    VN(objetivos_reales, predicciones),
    FP(objetivos_reales, predicciones),
    FN(objetivos_reales, predicciones)    
))

#%%

'''
###############################################################################
################                                          #####################
################          Métricas de evaluación          #####################
################                                          #####################
###############################################################################
''' 


'''Exactitud (accuracy)'''
metrics.accuracy_score(objetivos_reales, predicciones)

'''Precisión'''

def precision(clases_reales, predicciones):
    vp = VP(clases_reales, predicciones)
    fp = FP(clases_reales, predicciones)
    return vp / (vp+fp)

precision(objetivos_reales, predicciones)

'''Sensibilidad'''

metrics.recall_score(objetivos_reales, predicciones)

'''Puntuación F1'''

metrics.f1_score(objetivos_reales, predicciones)

#%%
