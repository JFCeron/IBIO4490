# entrenamiento y prueba de random forest usando los textones calculados
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import cv2

clf = RandomForestClassifier(n_estimators=1000, random_state=0)

train = np.load("train.npy")
train_labels = np.load("train_labels.npy")
test = np.load("test.npy")
test_labels = np.load("test_labels.npy")

# entrenar bosque aleatorio
clf.fit(train, train_labels)
# almacenar el modelo entrenado
pickle_out = open("randomForest.pickle","wb")
pickle.dump(clf, pickle_out)
pickle_out.close()

# matrices de confusion
from sklearn.metrics import confusion_matrix
train_confusion = confusion_matrix(train_labels, clf.predict(train)).astype("int")
test_confusion = confusion_matrix(test_labels, clf.predict(test)).astype("int")
np.savetxt("randomForest/train_confusion.txt", train_confusion)
np.savetxt("randomForest/test_confusion.txt", test_confusion)
# mostrar y guardar
img_confusion = cv2.resize((test_confusion*(255/np.max(test_confusion))).astype("uint8"), (200,200), interpolation=cv2.INTER_AREA)
plt.imshow(img_confusion)
cv2.imwrite("randomForest/test_confusion.png", img_confusion)