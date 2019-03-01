# -*- coding: utf-8 -*-
"""
extracts train and test sets, computes textons from training set, represents and stores both
in their texton representations and their labels
"""

import sys
sys.path.append('python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) # fbCreate(**kwargs, vis=True) for visualization

# images from disk
from skimage import color
from skimage import io
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
from assignTextons import assignTextons
from cifar10 import subsampling
# math
import numpy as np

# tamano del conjunto de entreno
n_train = 1500
# tamano del conjunto de prueba
n_test = 500
# number of clusters
k = 16*8
# numero de categorias
C = 10

cubo = subsampling(n_train,'Train')
cubo = np.array(cubo)
# imagenes de entreno a gris
cuboGris = color.rgb2gray(cubo.reshape((cubo.shape[0]*cubo.shape[1],cubo.shape[2],cubo.shape[3],cubo.shape[4])))
cuboGris = cuboGris.reshape((cubo.shape[0:4]))
N = cuboGris.shape[0]*cuboGris.shape[1]

# unir todas las imagenes del cubo
img_concatenada = np.zeros((32*C,32*int(N/C)))
for i in range(C):
    for j in range(int(N/C)):
        img_concatenada[32*i:32*(i+1),32*j:32*(j+1)] = cuboGris[i,j,:,:]

# aplicar filterbank
from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,img_concatenada)

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)
# map.shape = img_concatenada.shape
# textons.shape = (k,32)
#cuboTrain = np.zeros(cuboGris.shape)
from representar_con_textones import representar_con_textones

#train = cuboGris.reshape((N, cubo.shape[2], cubo.shape[3]))
train_labels = np.array([[i]*int(N/C) for i in range(C)]).flatten()
train = representar_con_textones(cuboGris, N, C, textons, fb)
#train = {'Imagenes':train,'labels':train_labels}

cuboTest = subsampling(n_test,'Test')
cuboTest = np.array(cuboTest)
cuboTest = color.rgb2gray(cuboTest.reshape((cuboTest.shape[0]*cuboTest.shape[1],cuboTest.shape[2],cuboTest.shape[3],cuboTest.shape[4])))
cuboTest = cuboTest.reshape((C, n_test, cuboTest.shape[1], cuboTest.shape[2]))
test_labels = np.array([[i]*cuboTest.shape[1] for i in range(C)]).flatten()
test = representar_con_textones(cuboTest, n_test*C, C, textons, fb)

#test = {'Imagenes':test,'labels':test_labels}
 
#almacenar textones y filter bank
np.save("textons",textons)
np.save("fb",fb)
np.save("train",train)
np.save("train_labels",train_labels)
np.save("test",test)
np.save("test_labels",test_labels)