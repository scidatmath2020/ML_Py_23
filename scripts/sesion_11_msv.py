# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:07:20 2023

@author: hp
"""

'''
###############################################################################
################                                          #####################
################      Máquinas de soporte vectorial       #####################
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
mi_data = pd.read_csv("datos_iris.csv")

#%%

variables_independientes = mi_data >> select(-_.Species)
variable_objetivo = mi_data >> select(_.Species) >> mutate(Species = _.Species.replace({"setosa":0,
                                                                                        "versicolor":1,
                                                                                        "virginica":2}))

variable_objetivo >> group_by(_.Species) >> summarize(total = n(_))

#%%

from sklearn.model_selection import train_test_split

iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(variables_independientes,
                                                                        variable_objetivo, test_size=0.3)

#%%

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

estimador_svm =  SVC()
estimador_svm.fit(iris_X_train, iris_y_train.values.ravel())
predicciones = estimador_svm.predict(iris_X_test)

f1_score(iris_y_test, predicciones, average="micro") #utilizamos micro porque las clases están balanceadas

cross_val_score(estimador_svm,
                X=variables_independientes,
                y=variable_objetivo.values.ravel(),
                cv=30,
                scoring="f1_micro").mean()

#%%

estimador_svm.support_vectors_
estimador_svm.n_support_

#%%
'''
kernels
'''
from mlxtend.plotting import plot_decision_regions

X = iris_X_train >> select(_.startswith("Sepal"))
y = iris_y_train

X.columns

estimador_svm_lineal = SVC(kernel="linear")
estimador_svm_lineal.fit(X=X,y=y.values.ravel())
cross_val_score(estimador_svm_lineal,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()

estimador_svm_poli2 = SVC(kernel="poly",degree=2)
estimador_svm_poli2.fit(X=X,y=y.values.ravel())
cross_val_score(estimador_svm_poli2,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()

estimador_svm_poli3 = SVC(kernel="poly",degree=3)
estimador_svm_poli3.fit(X=X,y=y.values.ravel())
cross_val_score(estimador_svm_poli3,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()

plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_lineal)
plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_poli2)
plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_poli3)

#%%
'''
kernel gaussiano
'''

estimador_svm_rbf = SVC(kernel="rbf")
estimador_svm_rbf.fit(X, y.values.ravel())
cross_val_score(estimador_svm_rbf,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()

plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_rbf)

estimador_svm_rbf_0_1 = SVC(kernel="rbf",gamma=0.1)
estimador_svm_rbf_0_1.fit(X, y.values.ravel())
plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_rbf_0_1)

estimador_svm_rbf_10 = SVC(kernel="rbf",gamma=10)
estimador_svm_rbf_10.fit(X, y.values.ravel())
plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_rbf_10)

estimador_svm_rbf_100 = SVC(kernel="rbf",gamma=100)
estimador_svm_rbf_100.fit(X, y.values.ravel())
plot_decision_regions(X.to_numpy(), y.values.ravel(), clf=estimador_svm_rbf_100)

cross_val_score(estimador_svm_rbf,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_0_1,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_10,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()
cross_val_score(estimador_svm_rbf_100,X=X,y=y.values.ravel(),cv=5,scoring="f1_micro").mean()



