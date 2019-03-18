# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:06:27 2019

@author: secg9
"""

def clusterImages(Img_file, method):
    from watershed import watershed
    from kmeans import kmeans
    import numpy as np
    import scipy.io as sio
    from skimage import io
    import pdb
    n_clusterings = 5
    img = io.imread('test/'+ Img_file)
    segmentaciones = {'segs': np.zeros((1,n_clusterings)).astype("object")}

    for i in range(2,n_clusterings+2):
        if "watershed" == method:
            clustering = watershed(img, i)
        if "kmeans" == method:
            clustering = kmeans(img, i)
        
        segmentaciones['segs'][0,i-2] = clustering

    sio.savemat(method+'2'+ '3' +'/'+ Img_file.replace('jpg', 'mat'),segmentaciones)
    
