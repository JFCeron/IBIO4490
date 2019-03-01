# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:56:00 2019

@author: secg9
"""

import pickle
from cifar10 import subsampling
import random
import numpy as np
from skimage import color
from assignTextons import assignTextons
from fbRun import fbRun
import cv2


cubo = subsampling(1,'Test')
cubo = random.sample(cubo, 3)
cubo = np.array(cubo)

cubo = cubo.reshape(cubo.shape[0]*cubo.shape[1],cubo.shape[2],cubo.shape[3],cubo.shape[4])

cuboGris = (color.rgb2gray(cubo)*255).astype('uint8')


fb =np.load('fb.npy')
textons = np.load('textons.npy')
cuboTextones = np.zeros(cuboGris.shape)

for i in range(cuboTextones.shape[0]):
   cuboTextones[i,:,:] = assignTextons(fbRun(fb,cuboGris[i,:,:]),textons.transpose())


with open('Knn.pickle', "rb") as input_file:
      knn = pickle.load(input_file)
       
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)   
     
      
k = len(textons)
HistogramasTest =[ histc(texton.flatten(), np.arange(k))/texton.size for texton in cuboTextones] 

Labels = knn.predict(HistogramasTest).astype("int")
    


font = cv2.FONT_HERSHEY_SIMPLEX

cuboTextones = cuboTextones.astype('uint8')

#int parameters tamaño,color,widht
ImagenesConEtiquetas = []
for i in range(0,len(cuboTextones)):
    ImagenesConEtiquetas.append(cv2.putText(cuboGris[i], str(Labels[i]), \
                (int(cuboTextones.shape[-1]/2),int(cuboTextones.shape[-2]/2)),font, 0.5,300,1))
    



vstack1 = np.vstack((cuboTextones[0],ImagenesConEtiquetas[0]))
vstack2 = np.vstack((cuboTextones[1],ImagenesConEtiquetas[1]))
vstack3 = np.vstack((cuboTextones[2],ImagenesConEtiquetas[2]))

hstack1 = np.hstack((vstack1,vstack2))
hstack2 = np.hstack((hstack1,vstack3))





cv2.imwrite('demo.jpg',hstack2)
cv2.imshow("frame1",hstack2) #display in windows 
cv2.waitKey(0) 
