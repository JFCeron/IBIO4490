# -*- coding: utf-8 -*-
"""
Load the model previously trained and depicts the confusion matrix over the test set
"""
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2

# cargar datos y labels de prueba
test = np.load("test.npy")
test_labels = np.load("test_labels.npy")

# cargar bosque aleatorio
clfFile = open("randomForest.pickle",'rb')
clf = pickle.load(clfFile)
# calcular y mostrar matriz de confusion
test_confusion = confusion_matrix(test_labels, clf.predict(test)).astype("int")
np.savetxt("randomForest/test_confusion.txt", test_confusion)
# mostrar y guardar
img_confusion = cv2.resize((test_confusion*(255/np.max(test_confusion))).astype("uint8"), (200,200), interpolation=cv2.INTER_AREA)
plt.imshow(img_confusion)
cv2.imwrite("randomForest/test_confusion.png", img_confusion)

# cargar Knn
knnFile = open("Knn.pickle",'rb')
neigh = pickle.load(knnFile)
# obtener histogramas del conjunto de prueba
def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)
textons = np.load('textons.npy')
k = len(textons)
HistogramasTest =[ histc(texton.flatten(), np.arange(k))/texton.size for texton in test] 
# calcular y mostrar matriz de confusion
test_confusion = confusion_matrix(test_labels, neigh.predict(HistogramasTest)).astype("int")
np.savetxt("Knn/test_confusion.txt", test_confusion)
# mostrar y guardar
img_confusion = cv2.resize((test_confusion*(255/np.max(test_confusion))).astype("uint8"), (200,200), interpolation=cv2.INTER_AREA)
plt.imshow(img_confusion)
cv2.imwrite("Knn/test_confusion.png", img_confusion)