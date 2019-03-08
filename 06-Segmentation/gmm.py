# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:08:42 2019

@author: secg9
"""


def gmm(img, k): 
    import time
    from sklearn import mixture
    import numpy as np
    #guarda la resolucion de la imagen
    resolucion = (img.shape[0],img.shape[1])
    #lista de vectores para aplicarle esta clusterizacion
    VectoresImagen = img.reshape((resolucion[0]*resolucion[1],img.shape[2]))
    #iplementacion de gmm
    Gmm = mixture.GaussianMixture(n_components = k)
    #se contabiliza el tiempo
    start_time = time.time()
    labelsGmm = Gmm.fit_predict(VectoresImagen)
    print("--- %s seconds ---" % (time.time() - start_time))
    #uso de la metrica
    ImgSegementadaGmm = labelsGmm.reshape((resolucion[0],resolucion[1]))
    return ImgSegementadaGmm   

