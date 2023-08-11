# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 16:58:59 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################             Preprocesamiento             #####################
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
mi_data = pd.read_csv("datos_procesamiento.csv")

#%%

mi_data.head()
mi_data.columns
mi_data.shape

#%%

'''
###############################################################################
################                                          #####################
################           Variables numéricas            #####################
################                                          #####################
###############################################################################
'''
#%%
'''Datos faltantes'''

from sklearn.impute import SimpleImputer


var_numericas_df = mi_data.select_dtypes([int, float])
var_numericas_df.columns

var_numericas_df[var_numericas_df.isnull().any(axis=1)]

imputador = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean")

var_numericas_imputadas = imputador.fit_transform(var_numericas_df)

var_numericas_imputadas_df = pd.DataFrame(var_numericas_imputadas,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_df[var_numericas_imputadas_df.isnull().any(axis=1)]

#%%
'''Estandarización'''

var_numericas_df.columns

var_numericas_imputadas_df.mean()
var_numericas_imputadas_df.std()


from sklearn import preprocessing

escalador = preprocessing.StandardScaler()
var_numericas_imputadas_escalado_standard = escalador.fit_transform(var_numericas_imputadas_df)

escalador.mean_
np.sqrt(escalador.var_)

var_numericas_imputadas_escalado_standard.mean(axis=0)
var_numericas_imputadas_escalado_standard.std(axis=0)

var_numericas_imputadas_escalado_standard_df = pd.DataFrame(var_numericas_imputadas_escalado_standard,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_standard_df

#%%

'''Escalado robusto'''

'''
Para aquellos casos en los que los datos tengan muchos valores extremos, es 
posible que estandarizar usando la media y la desviacion estandar no funcione 
bien en el modelo. Para esos casos es mejor usar unos estimadores mas robustos 
(menos sensibles a outliers) y emplear un RobustScaler que funciona substrayendo 
la mediana y escalando mediante el rango intercuartil (IQR).
'''

var_numericas_imputadas_df.columns

var_numericas_imputadas_df.mean(axis=0)


(ggplot(data = var_numericas_imputadas_df) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)

escalador_robusto = preprocessing.RobustScaler()
var_numericas_imputadas_escalado_robusto = escalador_robusto.fit_transform(
                                                        var_numericas_imputadas_df)

var_numericas_imputadas_escalado_robusto.mean(axis=0)
var_numericas_imputadas_escalado_robusto.std(axis=0)

var_numericas_imputadas_escalado_robusto_df = pd.DataFrame(var_numericas_imputadas_escalado_robusto,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_robusto_df


(ggplot(data = var_numericas_imputadas_escalado_robusto_df) +
     geom_boxplot(mapping=aes(x=1,y="col_outliers2"))
)
#%%
'''Escalado a un rango arbitrario'''

'''
Hay casos en los que en vez de estandardizar queremos escalar los datos a un 
rango (generalmente [-1,1] o [0,1]). Para ello podemos usar MinMaxScaler que 
hace escalado minmax (obviamente) o MaxAbscaler que simplemente divide cada valor 
de una variable por su valor máximo (y por tanto convierte el valor maximo a 1).
'''

var_numericas_imputadas_df.min()
var_numericas_imputadas_df.max()

'''
########################################## Escalador minmax
'''
escalador_minmax = preprocessing.MinMaxScaler()
var_numericas_imputadas_escalado_minmax = escalador_minmax.fit_transform(var_numericas_imputadas_df)

var_numericas_imputadas_escalado_minmax_df = pd.DataFrame(var_numericas_imputadas_escalado_minmax,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_minmax_df.min()
var_numericas_imputadas_escalado_minmax_df.max()

'''
########################################## Escalador maxabs
'''

escalador_maxabs = preprocessing.MaxAbsScaler()
var_numericas_imputadas_escalado_maxabs = escalador_maxabs.fit_transform(var_numericas_imputadas_df)

var_numericas_imputadas_escalado_minmax_df = pd.DataFrame(var_numericas_imputadas_escalado_maxabs,
                                                   index=var_numericas_df.index,
                                                   columns=var_numericas_df.columns)

var_numericas_imputadas_escalado_maxabs.max()


#%%

'''
###############################################################################
################                                          #####################
################           Variables categóricas          #####################
################                                          #####################
###############################################################################
'''
#%%

'''
Los modelos están diseñados para trabajar con variables numéricas.
Esto implica que para poder entrenar los modelos con variables categóricas
tenemos que convertirlas a números. Este proceso se llama codificación (encoding)
'''

mi_data.head()

var_categoricas = mi_data >> select(_.col_categorica,_.col_ordinal)

var_categoricas.head(10)


#%%

'''Variables ordinales'''

var_ordinales = mi_data >> select(_.col_ordinal)

var_ordinales >> n_distinct(_.col_ordinal)
var_ordinales["col_ordinal"].unique() 

label_codificador = preprocessing.OrdinalEncoder()
label_codificador.fit(var_ordinales)

label_codificador.transform(np.array([['muy bien'], ['muy mal'], ['muy bien'], ['muy mal'], ['bien'],["regular"]]))
label_codificador.inverse_transform([0, 0, 1, 2])


label_codificador.categories_

variables_ordinales_codificadas = label_codificador.fit_transform(var_ordinales)

var_ordinales_df = pd.DataFrame(variables_ordinales_codificadas,
                                                   index=var_ordinales.index,
                                                   columns=var_ordinales.columns)

var_ordinales_df


#%%

'''Variables nominales'''

'''Para variables nominales (por ejemplo animales) no tiene sentido usar este tipo de encoding'''

var_nominales = mi_data >> select(_.col_categorica)

oh_codificador = preprocessing.OneHotEncoder(sparse=False)
variables_nominales_codificadas = oh_codificador.fit_transform(var_nominales)

sorted(var_nominales["col_categorica"].unique())


var_nominales_df = pd.DataFrame(variables_nominales_codificadas,
                                                   index=var_nominales.index,
                                                   columns=sorted(var_nominales["col_categorica"].unique()))


#%%

'''
###############################################################################
################                                          #####################
################           Variables de texto             #####################
################                                          #####################
###############################################################################
'''

'''
Para convertir texto en variables numéricas, podemos proceder de igual forma 
que con las variables categóricas, simplemente separando las palabras antes.

Para ello tenemos dos vectorizadores en scikit-learn, que convierten texto en 
vectores.
'''
#%%

'''Contador'''


'''CountVectorizer devuelve un vector con el valor 0 en todas las palabras que 
no existen en una frase y con el numero de ocurrencias de las palabras que 
si existen
'''

from sklearn import feature_extraction

ejemplo_frases = ['los coches rojos',
          'los aviones son rojos',
          'los coches y los aviones son rojos',
          'los camiones rojos'
                 ]


vectorizador_count = feature_extraction.text.CountVectorizer()
X = vectorizador_count.fit_transform(ejemplo_frases)
vectorizador_count.get_feature_names_out()

pd.DataFrame(X.toarray(), columns=vectorizador_count.get_feature_names_out())

#%%

'''El tomar simplemente el número de veces que aparece cada palabra tiene un 
problema, y es que da un mayor peso a aquellas palabras que aparecen muchas 
veces pero que no aportan ningun valor semántico (por ejemplo, los). Una manera 
más sofisticada de vectorizar un texto es usar, en vez el número de apariciones, 
TF-IDF. TF-IDF se traduce como Frecuencia de Texto - Frecuencia Inversa de
Documento, y es una medida que asigna pesos a cada palabra en función de su 
frecuencia de aparición en todos los documentos.'''

vectorizador_tfidf = feature_extraction.text.TfidfVectorizer()
X = vectorizador_tfidf.fit_transform(ejemplo_frases)
pd.DataFrame(X.toarray(), columns=vectorizador_tfidf.get_feature_names_out())

#%%

vectorizador_tfidf = feature_extraction.text.TfidfVectorizer()
texto_vectorizado = vectorizador_tfidf.fit_transform(mi_data.col_texto)

pd.DataFrame(texto_vectorizado.toarray(), columns=vectorizador_tfidf.get_feature_names_out())

#%%

'''
###############################################################################
################                                          #####################
################               TODO JUNTO                 #####################
################                                          #####################
###############################################################################
'''

col_numericas =  ['col_inexistente1', 'col2', 'col3', 'col_outliers', 'col_outliers2']
col_ordinal = ['col_ordinal']
col_categorica = ['col_categorica']
col_texto = ['col_texto']

#%%
'''Variables numéricas'''
imputador = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean")
escalador = preprocessing.StandardScaler()
var_numericas_imputadas_escalado_standard = escalador.fit_transform(
                                                imputador.fit_transform(mi_data[col_numericas])
                                            )
df_numerico_procesado = pd.DataFrame(var_numericas_imputadas_escalado_standard,
                                                   columns=col_numericas)

#%%
'''Variables ordinales'''
label_codificador_ordinal = preprocessing.OrdinalEncoder()
variables_ordinales_codificadas = label_codificador_ordinal.fit_transform(mi_data[col_ordinal])

df_ordinal_procesado = pd.DataFrame(variables_ordinales_codificadas,
                                                   index=mi_data.index,
                                                   columns=mi_data[col_ordinal].columns)

df_ordinal_procesado

#%%
'''Variables nominales'''
label_codificador_nominal = preprocessing.OneHotEncoder(sparse=False)
variables_nominales_codificadas = label_codificador_nominal.fit_transform(mi_data[col_categorica])
df_nominal_procesado = pd.DataFrame(variables_nominales_codificadas, 
                                       columns=sorted(mi_data['col_categorica'].unique()))

#%%
'''Texto'''

vectorizador_tfidf = feature_extraction.text.TfidfVectorizer()
texto_vectorizado = vectorizador_tfidf.fit_transform(mi_data['col_texto'])
df_texto_procesado =  pd.DataFrame(texto_vectorizado.toarray(), columns=vectorizador_tfidf.get_feature_names_out())

#%%

'''Rearmando el dataframe'''

datos_procesados = pd.concat([
    df_numerico_procesado,
    df_ordinal_procesado,
    df_nominal_procesado,
    df_texto_procesado 
], axis=1)


datos_procesados.to_csv("datos_procesados.csv",index=False)












