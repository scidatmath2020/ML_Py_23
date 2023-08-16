# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:56:13 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################           Árboles de decisión            #####################
################                                          #####################
###############################################################################
'''

'''
Los algoritmos de creación de árboles están en el submódulo de sklearn.tree

En cuanto al tipo de algoritmo para crear árboles, scikit-learn usa una versión
optimizada del algoritmo CART (Classification and Regression Trees), que permite 
usar árboles de decisión tanto para problemas de clasificación como de regresión.
'''

import os
import pandas as pd
import numpy as np
from siuba import *




#%%

os.chdir("C:\\Users\\hp master\\Documents\\SciData\\23_ML_Py\\data\\")
mi_data = pd.read_csv("titanic.csv")

mi_data.head()

#%%

from sklearn import tree
from sklearn.model_selection import cross_val_score

#%%

columnas_categoricas = ["genero", "puerto_salida"]
datos_categoricos = pd.get_dummies(mi_data[columnas_categoricas])

#%%
pasajeros = (
    pd.concat([
        mi_data.drop(columnas_categoricas, axis=1),
        datos_categoricos
    ],axis=1
    )
)

#%%
pasajeros.edad = pasajeros.edad.fillna(pasajeros.edad.mean())

#%%

variables_independientes = pasajeros >> select(-_.superviviente)
variable_objetivo = pasajeros >> select(_.superviviente)


#%%
arbol = tree.DecisionTreeClassifier() 
arbol.fit(variables_independientes, variable_objetivo)

#%%
cross_val_score(arbol,
                variables_independientes, 
                variable_objetivo, 
                scoring="roc_auc", 
                cv=40).mean()

#%%

os.environ["PATH"] = os.pathsep + "C:\\Program Files\\Graphviz\\bin"

def dibujar_arbol(arbol):
    dot_data = tree.export_graphviz(arbol, out_file=None,
                                    feature_names=pasajeros.drop("superviviente",axis=1).columns,
                                    class_names = pasajeros.superviviente.astype(str),
                                    filled=True
                                    )
    graph = graphviz.Source(dot_data)
    graph.format = "png"
    graph.render("arbol",view=True)

dibujar_arbol(arbol)

#%%

'''
Se puede exportar el árbol y abrirlo posteriormente con graphviz desde la terminal
(o desde la página http://webgraphviz.com/ que renderiza archivos de graphviz)
'''

tree.export_graphviz(arbol, out_file="arbol.dot",
                     feature_names=pasajeros.drop("superviviente",axis=1).columns,
                     class_names = pasajeros.superviviente.astype(str),
                     filled=True)

#%%

'''Podemos ver también cuáles son las variables más importantes'''

dict(zip(
    pasajeros.drop("superviviente", axis=1),
    arbol.feature_importances_
))


sorted(zip(
    arbol.feature_importances_,
    pasajeros.drop("superviviente", axis=1)), 
    reverse = True)

#%%

variables_independientes.iloc[0]

pasajero_nvo = pd.DataFrame(
    {'clase_billete': [3.],
     'edad': [22.],
     'n_hermanos_esposos': [1.],
     'n_hijos_padres': [0.],
     'precio_billete': [7.25],
     'genero_hombre': [1.],
     'genero_mujer': [0.],
     'puerto_salida_C': [0.],
     'puerto_salida_Q': [0.],
     'puerto_salida_S': [1.]
     }
    )

arbol.predict(pasajero_nvo)

#%%
'''
Éstos son los parámetros más importantes para los modelos DecisionTreeClassifier 
de sklearn:

criterion : El criterio para calcular la reducción de impureza (ganancia de información) 
al hacer una partición. Se puede elegir entre gini, o entropy

max_depth : La profundidad máxima del árbol. Definimos profundidad como el número 
de nodos que atraviesa una observación (cuantas "preguntas" se le hacen).

max_features: El máximo numero de particiones potenciales que se consideran al 
evaluar un nodo.

max_leaf_nodes : Límite de hojas para el árbol.

min_impurity_decrease : la ganancia de información mínima en un nodo para hacer 
una partición. (Si no hay ninguna partición que cumpla este criterio, se para el 
                desarrollo del árbol en dicho nodo).

class_weight : Para clases imbalanceadas, podemos pasar el argumento class_weight, 
como un diccionario de {clase: peso} para que sklearn tenga en cuenta los pesos. 
Alternativamente, podemos pasar el string balanced para que sklearn genere pesos 
en función del número de muestras de cada clase.
'''

'''
Control del máximo de profundidad
'''

arbol_simple = tree.DecisionTreeClassifier(max_depth=3)
arbol_simple.fit(variables_independientes, variable_objetivo)
cross_val_score(arbol_simple,
                variables_independientes, 
                variable_objetivo, 
                scoring="roc_auc", 
                cv=40).mean()

dibujar_arbol(arbol_simple)

for k in range(3,21):
    arbol_simple = tree.DecisionTreeClassifier(max_depth=k)
    arbol_simple.fit(pasajeros.drop("superviviente", axis=1), pasajeros.superviviente)
    calificacion = cross_val_score(arbol_simple, pasajeros.drop("superviviente", axis=1), 
                pasajeros.superviviente, scoring="roc_auc", cv=10).mean()
    print(f"{(k,calificacion)}")

#%%
'''
balanceo
'''

arbol_balanceado = tree.DecisionTreeClassifier(max_depth=3, class_weight="balanced")
arbol_balanceado.fit(pasajeros.drop("superviviente", axis=1), pasajeros.superviviente)
dibujar_arbol(arbol_balanceado)

cross_val_score(arbol_balanceado, pasajeros.drop("superviviente", axis=1), 
                pasajeros.superviviente, scoring="roc_auc", cv=10).mean()

#%%

'''
Además del algoritmo CART para generar árboles, scikit-learn también proporciona 
una clase de arboles llamada ExtraTreeClassifier, o Extremely Random Trees 
(Árboles Extremadamente Aleatorios). En estos árboles, en lugar de seleccionar 
en cada nodo la párticion que proporciona la mayor ganancia de información, 
¡se decide una partición al azar!.
'''

arbol_aleatorio = tree.ExtraTreeClassifier(max_features=1)
arbol_aleatorio.fit(pasajeros.drop("superviviente", axis=1), pasajeros.superviviente)

dibujar_arbol(arbol_aleatorio)

cross_val_score(arbol_aleatorio, pasajeros.drop("superviviente", axis=1), 
                pasajeros.superviviente, scoring="roc_auc",
                cv=10).mean()




    
    
    








