# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time
from sklearn.metrics import precision_recall_curve
import cv2
import os
from copy import deepcopy
import random
import pdb
from sklearn.metrics import confusion_matrix
def sigmoid(x):
    one = 0.9999
    zero = 0.0001
    retorno = 1/(1+np.exp(-x))
    retorno[retorno == 1] = one
    retorno[retorno == 0] = zero
    return retorno

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
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    # pull a validation set from the train set matching test set size
    val_indices = np.random.choice(range(len(y_train)), size=len(y_test), replace=False)
    y_val = y_train[val_indices]
    y_train = np.delete(y_train, val_indices)
    x_val = x_train[val_indices,:]
    x_train = np.delete(x_train, val_indices, axis=0)

    x_train /= 255 #normalize inputs between [0, 1]
    x_val /= 255
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_val = x_val.reshape(x_val.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.01 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)
        self.train_time = 0

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr

def train(model):
    # save total training time
    start_time = time.time()
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    batch_size = 100 # Change if you want
    epochs = 10000  # Change if you want
    losses = np.zeros((2,epochs))
    #model with minimum validation error
    final_model= deepcopy(model)
    for i in range(epochs):
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            model.compute_gradient(_x_train, out, _y_train)
        # validation and training loss in this epoch
        # estimate train loss on val-sized subset of train
        outTrain = model.forward(x_train)
        outVal = model.forward(x_val)
        losses[0,i] = model.compute_loss(outTrain, y_train)
        losses[1,i] = model.compute_loss(outVal, y_val)
        # this might be the lowest validation error
        if i>0 and losses[1,i]==np.min(losses[1,:(i+1)]):
            final_model = deepcopy(model)
        if i % 20 == 0:
            print("Epoch "+str(i)+", lr="+str(model.lr)+": Train loss="+str(losses[0,i])+", Validation loss="+str(losses[1,i]))
            # store total computation time, losses and current best model
            final_model.train_time = time.time() - start_time
            np.save("losses/logistic_lr"+str(model.lr)+".npy", np.array(losses))
            # pickle the trained model object
            model_file = open("models/logistic_lr"+str(model.lr)+".obj", "wb")
            pickle.dump(final_model, model_file)
    model = final_model

def plot(train_loss, test_loss):
    assert train_loss.shape==test_loss.shape,  "Different length matrices provided."
    # set color scales for train (blue) and test (red) set
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
       ax.plot(x_ticks, test_loss[i,:], c=cmap_test.to_rgba(i + 1))
    plt.gca().legend(('Train loss, small lr','Test loss, small lr'))
    plt.show()
    # TODO guardar imagen en pdf

def test(model):
    x_train, y_train, x_val, y_val, x_test, y_test = get_data()
    y_pred = np.zeros((x_test.shape[0],1)) 
    y_pred[model.forward(x_test) < 0] = 1
    precision, recall, thresholds = precision_recall_curve(y_test, sigmoid(model.forward(x_test)))    
    # plot ROC curve
    plt.plot(recall, precision)
    plt.savefig('PRcurve_model_lr' + str(model.lr) + '.jpg')
    # and report goodness measures
    precision[precision == 0] = 0.0001
    recall[recall == 0] = 0.001
    Fmeasure = (2*precision*recall)/(precision + recall)
    FmeasureMax = Fmeasure.max()
    pred = model.forward(x_test)
    y_hat = np.argmax(pred, axis=1)
    confusion = confusion_matrix(y_hat, y_test)
    print("Best ACA: " + str(np.sum(np.diag(confusion))/np.sum(confusion)))
    print("Max F1-measure: "+ str(FmeasureMax))

if __name__ == '__main__':
    model = Model()
    if "--test" in sys.argv:
        pickle_off = open('./models/logistic_lr1e-05.obj',"rb")
        model = pickle.load(pickle_off)
        test(model)
    elif "--demo" in sys.argv:

        modelos = os.listdir('./models/')
        pickle_off = open('./models/logistic_lr1e-05.obj',"rb")
        model = pickle.load(pickle_off)
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
        imageTag =  sigmoid(model.forward(ListaImagenes))
        for i in range(0,6):
            text = ''
            if (imageTag[i] < 0.5):
                   text = 'No feliz'
            else:
                   text = 'feliz' 
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
        if('fer2013.csv' not in os.listdir('.')):
             os.system('wget https://www.dropbox.com/s/n8wen5fbzdm9ujy/fer2013.csv?dl=0')
        train(model)
        test(model)
