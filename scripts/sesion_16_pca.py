# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 19:18:31 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################         Componentes principales          #####################
################                                          #####################
###############################################################################
'''

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import *
from plotnine import *
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("cancer_mama.csv")

#%%

variables_independientes = mi_data >> select(-_.diagnosis)
variable_objetivo = mi_data >> select(_.diagnosis)

#%%

from sklearn import preprocessing
escalador = preprocessing.StandardScaler()
variables_escaladas = escalador.fit_transform(variables_independientes)

varriables_escaladas = pd.DataFrame(variables_escaladas,
                                    index = variables_independientes.index,
                                    columns = variables_independientes.columns)

variables_escaladas.shape

#%%

from sklearn.decomposition import PCA

modelo_pca = PCA(n_components = 2)
componentes_principales = modelo_pca.fit_transform(variables_escaladas)

componentes_principales.shape 

componentes_df = pd.DataFrame(componentes_principales,columns = ["comp1","comp2"])

(componentes_df >> mutate(diagnostico = mi_data["diagnosis"].astype(str)) >>
    ggplot() +
    geom_point(mapping = aes(x="comp1",y="comp2",color = "diagnostico"),alpha = 0.5)
)

modelo_pca.explained_variance_
sum(modelo_pca.explained_variance_ratio_)



#%%

'''¿Cuantos componentes seleccionar?
En cuanto al número de componentes que elegir, podemos seguir ciertas reglas de
sentido común:

- Elegir al menos componentes que sumen el 80% de la varianza total

- Realizar un gráfico de los valores propios de cada componente de forma 
decreciente y usar el método del codo para ver en que momento hay ganancias 
decrecientes de los componentes principales (esto se llama gráfica SCREE).'''


modelos_pca = PCA() 
modelos_pca.fit_transform(variables_escaladas)

resultados = pd.DataFrame()
np.cumsum(modelos_pca.explained_variance_ratio_)

percent_varianza = pd.DataFrame({"n_comps" : list(range(1,variables_escaladas.shape[1]+1)),
                          "var_acumulada" : np.cumsum(modelos_pca.explained_variance_ratio_)})

(ggplot(data = percent_varianza) +
     geom_point(mapping = aes(x="n_comps",y="var_acumulada")) +
     geom_line(mapping = aes(x="n_comps",y="var_acumulada"))
)

varianzas = pd.DataFrame({"n_comps" : list(range(1,variables_escaladas.shape[1]+1)),
                          "var" : modelos_pca.explained_variance_})

(ggplot(data = varianzas) +
     geom_point(mapping = aes(x="n_comps",y="var")) +
     geom_line(mapping = aes(x="n_comps",y="var"))
)

#%%











