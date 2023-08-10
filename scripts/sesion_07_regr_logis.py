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

mi_data

#%%

mi_data.shape

mi_data.columns

#%%

mi_data >> group_by(_.diagnosis) >> summarize(conteo_objetivo = n(_), porcentaje_objetivo = n(_)/mi_data.shape[0])

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
                         

modelo_reg_logis.predict_proba(data_peor_area >> select(_.worst_area))

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


modelo_rl = LogisticRegression(solver = "liblinear")

modelo_rl.fit(indepen_entrenamiento,objetivo_entrenamiento.values.ravel())

predicciones = modelo_rl.predict(indepen_prueba)
predicciones_probabilidades = modelo_rl.predict_proba(indepen_prueba)
objetivos_reales = objetivo_prueba.values.ravel()

#%%

def tupla_clase_prediccion(y_real, y_pred):
    return list(zip(y_real, y_pred))

tupla_clase_prediccion(objetivos_reales, predicciones)[:30]


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

pd.DataFrame({"exactitud":[metrics.accuracy_score(objetivos_reales, predicciones)],
 "precision":[precision(objetivos_reales, predicciones)],
 "sensibilidad":[metrics.recall_score(objetivos_reales, predicciones)],
 "F1":[metrics.f1_score(objetivos_reales, predicciones)]
})


#%%

def proba_a_etiqueta(predicciones_probabilidades,umbral=0.5):
    predicciones = np.zeros([len(predicciones_probabilidades), ])
    predicciones[predicciones_probabilidades[:,1]>=umbral] = 1
    return predicciones

proba_a_etiqueta(predicciones_probabilidades)


#%%

def evaluar_umbral(umbral):
    predicciones_en_umbral = proba_a_etiqueta(predicciones_probabilidades, umbral)
    precision_umbral = precision(objetivos_reales, predicciones_en_umbral)
    sensibilidad_umbral = metrics.recall_score(objetivos_reales, predicciones_en_umbral)
    F1_umbral = metrics.f1_score(objetivos_reales, predicciones_en_umbral)
    return (umbral,precision_umbral, sensibilidad_umbral, F1_umbral)

#%%
umbrales = np.linspace(0., 1., 1000)

evaluaciones = pd.DataFrame([evaluar_umbral(x) for x in umbrales],
                            columns = ["umbral","precision","sensibilidad","F1"])

#%%

(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="sensibilidad",y="precision",color="umbral"),size=0.1)
)

(ggplot(data = evaluaciones) +
    geom_point(mapping=aes(x="umbral",y="F1"),size=0.1)
)

evaluaciones >> filter(_.F1 == _.F1.max())




