# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time
import cv2
import os
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import random
import pdb
# cast the rows of M to probabilities
def softmax(M):
    e_power = np.exp(M)
    return e_power / np.sum(e_power, axis=1).reshape((M.shape[0],1))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1,num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        # emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'int')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'int')

    # pull a validation set from the train set matching test set size
    val_indices = np.random.choice(range(len(y_train)), size=len(y_test), replace=False)
    y_val = y_train[val_indices]
    y_train = np.delete(y_train, val_indices)
    x_val = x_train[val_indices,:]
    x_train = np.delete(x_train, val_indices, axis=0)

    x_train /= 255 #normalize inputs between [0, 1]
    x_val /= 255
    x_test /= 255

    #x_train = x_train.reshape(x_train.shape[0], 48, 48)
    #x_val = x_val.reshape(x_val.shape[0], 48, 48)
    #x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    return x_train.reshape(x_train.shape[0],-1), y_train, x_val.reshape(x_val.shape[0],-1), y_val, x_test.reshape(x_test.shape[0],-1), y_test

class Model():
    def __init__(self):
        self.C = 7 # number of classes
        self.params = 48*48 # pixels * number of classes
        self.lr = 0.01 # Change if you want
        self.W = np.random.randn(self.params, self.C)
        self.b = np.random.randn(1, self.C)
        self.train_time = 0

    def forward(self, image):
        # image = image.reshape(image.shape[0], -1)
        out = softmax(np.dot(image, self.W) + self.b)
        return out

    def compute_loss(self, pred, gt):
        # pred has vectors of probabilities for rows
        # true labels to convenient binary matrix
        B = np.zeros((gt.shape[0],self.C))
        for i in range(len(gt)): B[i,gt[i,0]] = 1
        J = np.sum(np.ma.log(np.multiply(pred, B)))*(-1/pred.shape[0])
        return J

    def compute_gradient(self, image, pred, gt):
        # different dimensions in the matrices
        F = image.shape[1]
        C = self.C
        N = image.shape[0]
        # same convenient matrix
        B = np.zeros((N,C))
        for i in range(N): B[i,gt[i,0]] = 1

        # calculate W gradient
        B = B.reshape(1, 1, N, C) # Dirac i=y(x)
        X = np.transpose(image).reshape(F, 1, N, 1) # data
        P = np.transpose(pred).reshape(1, C, N, 1) # predicted
        M = np.identity(C).reshape(1, C, 1, C) # Dirac i=c
        W_grad = -np.sum(B*(X*(M-P)), axis=(2,3))/N
        # and b gradient
        B = B.reshape(1, N, C)
        P = P.reshape(C, N, 1)
        M = M.reshape(C, 1, C)
        b_grad = -np.sum(B*(M-P), axis=(1,2))/N

        # finally update
        self.W -= W_grad*self.lr
        self.b -= b_grad*self.lr

def train(model):
    # start recording time
    start_time = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    batch_size = 400 # Change if you want
    epochs = 20000  # Change if you want
    # first row: train losses, second : val losses
    losses = np.zeros((2,epochs))
    # model with minimum validation set error
    final_model = deepcopy(model)
    for i in range(epochs):
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            model.compute_gradient(_x_train, out, _y_train)
        # validation and training loss in this epoch
        # estimate train loss on val-sized subset of train
        train_subset = np.random.choice(range(len(x_train)), size=len(x_val), replace=False)
        losses[0,i] = model.compute_loss(model.forward(x_train[train_subset]), y_train[train_subset])
        losses[1,i] = model.compute_loss(model.forward(x_val), y_val)
        # this might be the lowest validation error
        if i>0 and losses[1,i]==np.min(losses[1,:i]):
            final_model = deepcopy(model)
        if i % 20 == 0:
            print("Epoch "+str(i)+", lr="+str(model.lr)+": Train loss="+str(losses[0,i])+", Validation loss="+str(losses[1,i]))
            # store total computation time, losses and current best model
            final_model.train_time = time.time() - start_time
            np.save("losses/softmax_lr"+str(model.lr)+".npy", np.array(losses))
            # pickle the trained model object
            model_file = open("models/softmax_lr"+str(model.lr)+".obj", "wb")
            pickle.dump(final_model, model_file)
    model = final_model


def plot(train_loss, val_loss):
    assert train_loss.shape==val_loss.shape,  "Different length matrices provided."

    # set color scales for train (blue) and validation (red) set
    norm = mpl.colors.Normalize(vmin=0, vmax=train_loss.shape[0])
    cmap_train = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap_train.set_array([])
    cmap_test = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
    cmap_test.set_array([])
    # one x tick per epoch
    x_ticks = range(train_loss.shape[1])

    # plot train and test losses at R learning rates across N epochs
    fig, ax = plt.subplots(dpi=100)
    for i in range(train_loss.shape[0]):
       ax.plot(x_ticks, train_loss[i,:], c=cmap_train.to_rgba(i + 1))
       ax.plot(x_ticks, val_loss[i,:], c=cmap_test.to_rgba(i + 1))
    plt.gca().legend(('Train loss','Val loss'))
    plt.show()
    # TODO guardar imagen en pdf

def test(model):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    pred = model.forward(x_test)
    y_hat = np.argmax(pred, axis=1)
    confusion = confusion_matrix(y_hat, y_test)
    print("Best ACA: " + str(np.sum(np.diag(confusion))/np.sum(confusion)))

if __name__ == '__main__':
    if "--test" in sys.argv:
        pickle_in = open("models/softmax_lr0.01.obj","rb")
        model = pickle.load(pickle_in)
        test(model)
    elif "--demo" in sys.argv:
        modelos = os.listdir('./models/')
        #pickle_off = open("models/softmax_lr0.01.obj","rb")
        #model = pickle.load(pickle_off)
        model = Model()
        archivos = os.listdir('./in-the-wild/')
        imagenesMostrar = random.sample(archivos, 6)
        font = cv2.FONT_HERSHEY_SIMPLEX


        ListaImagenes = []
        ListaIamgenesParaMostrar = []
        for file in imagenesMostrar: 
            img = cv2.imread('./in-the-wild/' + file)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(48,48))
            aMostrar= cv2.resize(img,(256,256))
            ListaImagenes.append(gray)
            ListaIamgenesParaMostrar.append(aMostrar)

        ListaImagenes = np.array(ListaImagenes)
        ListaIamgenesParaMostrar = np.array(ListaIamgenesParaMostrar)
        ListaImagenes = ListaImagenes /  255
        imageTag =  model.forward(ListaImagenes.reshape(ListaImagenes.shape[0],-1))

        for i in range(0,6):
            text = ''
            emocion = np.argmax(imageTag[i])
            if (emocion ==0):
                   text = 'angry'
            if(emocion == 1):
                   text = 'Disgust'
            if(emocion == 3):
                   text = 'Fear'
            if(emocion == 4):
                   text = 'Happy'
            if(emocion == 5):
                   text = 'Sad'
            if(emocion == 6):
                   text = 'Surprise'
            if(emocion == 7):
                   text = 'Neutral'
            ListaIamgenesParaMostrar[i] = cv2.putText(ListaIamgenesParaMostrar[i], text, (40,40),font,1,200,2)

        vstack1 = np.vstack((ListaIamgenesParaMostrar[0],ListaIamgenesParaMostrar[1]))
        vstack2 = np.vstack((ListaIamgenesParaMostrar[2],ListaIamgenesParaMostrar[3]))
        vstack3 = np.vstack((ListaIamgenesParaMostrar[4],ListaIamgenesParaMostrar[5]))

        hstack1 = np.hstack((vstack1,vstack2))
        hstack2 = np.hstack((hstack1,vstack3))


        cv2.imshow("frame1",hstack2) #display in windows 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if ('fer2013.csv' not in os.listdir('.')):
               os.system('wget https://www.dropbox.com/s/n8wen5fbzdm9ujy/fer2013.csv?dl=0')

        model = Model()
        train(model)
        test(model)
