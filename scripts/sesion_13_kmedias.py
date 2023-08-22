# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:13:19 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################                k medias                  #####################
################                                          #####################
###############################################################################
'''

import os
import pandas as pd
import numpy as np
from siuba import *
from siuba.dply.vector import * 
from plotnine import *
import plotly.express as px
from plotly.offline import plot

from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer

#%%

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("datos_clientes.csv")

#%%
mi_data = mi_data >> select(-_.Id_cliente) >> mutate(Genero = _.Genero.replace({"Female":1,"Male":0})) 

escalador = preprocessing.normalize(mi_data)
mi_data_normalizado_df = pd.DataFrame(escalador, 
                                      index=mi_data.index, 
                                      columns=mi_data.columns)

#%%

from sklearn.cluster import KMeans

modelos = KMeans(n_init=10)

visualizer = KElbowVisualizer(modelos, k=(1,12))
visualizer.fit(mi_data_normalizado_df)
visualizer.show()

#%%

k_medias = KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan')
k_medias.fit(mi_data_normalizado_df)
k_medias.labels_
k_medias.cluster_centers_
#k_medias.predict()

'''
parámetros:
    n_clusters: número de clusters
    init: 'k-means++', 'random' o arreglo de tamaño (n_clusters,n_características)
    n_init: 'auto' o entero. Para correr diferentes inicializaciones (1 si init=k-means++ o 10 si 
                                                                      init = random o arreglo)
    max_iter: entero (default = 300); máximo número de iteraciones    
'''


mi_data = mi_data >> mutate(Etiquetas = k_medias.labels_.astype(str))

fig = px.scatter_3d(mi_data, x='Edad', y='Puntuacion_gasto', z='Ingreso_anual',
              color='Etiquetas')
plot(fig)






