# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:35:39 2019

@author: secg9
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2

textons = np.load('textons.npy')
k = len(textons)

train = np.load('train.npy')
train = train.reshape(train.shape[0],int(train.shape[1]/32),int(train.shape[1]/32))
train_labels = np.load('train_labels.npy')
test = np.load('test.npy')
test_labels = np.load('test_labels.npy')
test = test.reshape(test.shape[0],int(test.shape[1]/32),int(test.shape[1]/32))

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

Histogramas =[ histc(texton.flatten(), np.arange(k))/texton.size for texton in train]
HistogramasTest =[ histc(texton.flatten(), np.arange(k))/texton.size for texton in test] 

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(Histogramas, train_labels)
# almacenar el modelo entrenado
pickle_out = open("Knn.pickle","wb")
pickle.dump(neigh, pickle_out)
pickle_out.close()

from sklearn.metrics import confusion_matrix
test_confusion = confusion_matrix(test_labels, neigh.predict(HistogramasTest)).astype("int")
np.savetxt("Knn/test_confusion.txt", test_confusion)
# mostrar y guardar
img_confusion = cv2.resize((test_confusion*(255/np.max(test_confusion))).astype("uint8"), (200,200), interpolation=cv2.INTER_AREA)
plt.imshow(img_confusion)
cv2.imwrite("Knn/test_confusion.png", img_confusion)