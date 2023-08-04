# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:11:11 2023

@author: hp
"""

import os
import pandas as pd
import numpy as np
from plotnine import *
from siuba import *


#%%
ruta = "C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py"
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













