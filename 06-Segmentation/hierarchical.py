# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:12:00 2019

@author: secg9
"""


def  hierarchical(img, k):
    import time
    import sklearn
    from sklearn.cluster import KMeans
    import numpy as np
    #guarda la resolucion
    resolucion = (img.shape[0],img.shape[1])
    #crea un vector para los metodos que clusterizan
    VectoresImagen = img.reshape((resolucion[0]*resolucion[1],img.shape[2]))
    
    #clusterizacion
    start_time = time.time()
    kmeans = KMeans(n_clusters = 10, random_state=0).fit(VectoresImagen)
    print("--- %s seconds ---" % (time.time() - start_time))
    #se guardan los centroides
    centroids  = kmeans.cluster_centers_.astype(int)
    #se guardan los labesl del knn con el fin de hacer un aimagen con los centroides
    labels = kmeans.labels_
    labels = labels.reshape((resolucion[0],resolucion[1]))
    
    #se arma la imagen con los centroides
    temp = img.copy()
    temp[:,:] = centroids[labels[:,:]]
    
    
    #son los centroides los que se clusterizan de manera jerarquicos
    jerarquico = sklearn.cluster.AgglomerativeClustering(n_clusters=k,linkage='average'  )
    Clusters = jerarquico.fit(centroids)
    
    #labels de la clusterizacion
    labelsFinal = Clusters.labels_ 
    
    #vector de centroides con sus labels al hacer el clustering jerarquico    
    vectores = np.zeros((centroids.shape[0],4))    
    vectores[:,0:3] = centroids
    vectores[:,-1] = labelsFinal
    
   
    #se arma la imagen con clustering jerarquico    
    temp2 = np.zeros((resolucion[0],resolucion[1]))    
    for i in range(0,temp2.shape[0]):
        for j in range(0,temp2.shape[1]):
            temp2[i,j] = vectores[np.argwhere(vectores[:,0:3] == temp[i,j])[0],3][0]
            
    return temp2