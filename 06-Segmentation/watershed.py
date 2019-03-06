# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:06:13 2019

@author: Bananin
"""

def watershed (img, k=5):
    import numpy as np
    from skimage.filters import rank
    from skimage.morphology import watershed, disk
    from scipy import ndimage as ndi
    # analizamos el gradiente canal por canal
    # remover ruido
    denoised = img.copy()
    gradient4markers = img.copy()
    gradient4watershed = img.copy()
    for i in range(img.shape[-1]):
        # -> float para que los gradientes sean suaves
        denoised[...,i] = rank.median(img[...,i], disk(3)).astype('float')
        gradient4markers[...,i] = rank.gradient(denoised[...,i], disk(5))
        gradient4watershed[...,i] = rank.gradient(denoised[...,i], disk(2))
    
    # agregamos los gradientes de cada canal (promedio)
    gradient4markers = np.mean(gradient4markers, axis=2)
    gradient4watershed = np.mean(gradient4watershed, axis=2)
    # definimos marcadores como zonas de bajo gradiente relativo    
    markers = gradient4markers <= np.percentile(gradient4markers, q=0.1)
    markers = ndi.label(markers)[0]
    # admitimos un maximo de k clusters; tomar los k mas grandes
    if np.max(markers)>k:
        marker_labels = np.unique(markers[markers > 0])
        sizes = np.array([np.sum(markers == i) for i in marker_labels])
        labels_sorted = marker_labels[sizes.argsort()[::-1]]
        labels_retained = labels_sorted[0:k]
        markers[np.logical_not(np.isin(markers, labels_retained))] = 0
    
    # process the watershed
    labels = watershed(gradient4watershed, markers)
    return(labels)