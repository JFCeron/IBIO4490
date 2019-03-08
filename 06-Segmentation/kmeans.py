# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:08:41 2019

@author: secg9
"""

def kmeans(img, k):
    import numpy as np
    from sklearn.cluster import KMeans
    resolucion = (img.shape[0],img.shape[1])
    VectoresImagen = img.reshape((resolucion[0]*resolucion[1],img.shape[2]))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(VectoresImagen)
    labelsKnn = kmeans.labels_
    ImgSegementadaKnn= labelsKnn.reshape((img.shape[0],img.shape[1]))
    return ImgSegementadaKnn
    
    
    