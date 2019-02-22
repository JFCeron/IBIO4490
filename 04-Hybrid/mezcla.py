# manipulacion de imagenes
from PIL import Image
import cv2
import matplotlib.pyplot as plt
# matematicas
import numpy as np
# imagenes para mezclar
dante = Image.open("dante.jpg")
pato = Image.open("whiteduck.jpg")
# alinear imagenes
pato = pato.rotate(-2).crop((100,80,pato.size[0]-75,pato.size[1]-30))
# ajustar tamanos
tamano = (int(np.mean([dante.size[0],pato.size[0]])),int(np.mean([dante.size[1],pato.size[1]])))
dante = dante.resize(tamano)
pato = pato.resize(tamano)
# almacenar foto editada del pato
plt.imshow(pato).write_png('pato_modificado.jpg')
# convertir a matriz
dante = np.array(dante)
pato = np.array(pato)
# filtar imagenes
blurrpato = 100
blurrdante = 10
lowpass = cv2.GaussianBlur(dante, ksize=(51,51), sigmaX=blurrdante, sigmaY=blurrdante).astype('int')
highpass = pato - cv2.GaussianBlur(pato, ksize=(51,51), sigmaX=blurrpato, sigmaY=blurrpato).astype('int')
highpass[highpass < 0] = 0
# imagen hibrida
hibrida = highpass+lowpass
hibrida[hibrida > 255] = 255
hibrida = hibrida.astype('uint8')
plt.imshow(hibrida).write_png('danteduck.jpg')

# piramide
altura = 5
espacio = 10
piramide = np.zeros((2*hibrida.shape[0] + espacio*altura,hibrida.shape[1],3)).astype('uint8')+255
piramide[0:hibrida.shape[0],:,:] = hibrida
zoom_actual = hibrida
y_actual = hibrida.shape[0]+espacio
for i in range(1,altura):
    zoom_actual = cv2.pyrDown(zoom_actual)
    piramide[y_actual:(y_actual+zoom_actual.shape[0]), 0:zoom_actual.shape[1],:] = zoom_actual
    y_actual = y_actual+zoom_actual.shape[0]+espacio
plt.imshow(piramide).write_png('piramide.jpg')

# blended: construccion de piramides gaussianas y laplacianas
G_dante = []
L_dante = []
G_pato = []
L_pato = []
dante_actual = dante
pato_actual = pato
for i in range(5):
    G_dante.append(dante_actual)
    G_pato.append(pato_actual)
    dante_actual = cv2.pyrDown(dante_actual)
    pato_actual = cv2.pyrDown(pato_actual)
    L_i_dante = G_dante[i].astype('int') - cv2.pyrUp(dante_actual)[0:G_dante[i].shape[0],0:G_dante[i].shape[1],:].astype('int')
    L_i_pato = G_pato[i].astype('int') - cv2.pyrUp(pato_actual)[0:G_pato[i].shape[0],0:G_pato[i].shape[1],:].astype('int')
    L_i_dante[L_i_dante < 0] = 0
    L_i_pato[L_i_pato < 0] = 0
    L_dante.append(L_i_dante.astype('uint8'))
    L_pato.append(L_i_pato.astype('uint8'))

# concatenacion de laplacianas
concat = []
for i in range(5):
    concat_i = L_dante[i]
    concat_i[:,0:int(concat_i.shape[1]/2),:] = L_pato[i][:,0:int(concat_i.shape[1]/2),:]
    concat.append(concat_i)

# reconstruccion de imagen blended
blended = concat[4]
for i in range(4):
    blended = cv2.pyrUp(blended)
    if concat[3-i].shape[1]%2 == 1:
        blended = cv2.add(blended[:,0:(blended.shape[1]-1),:],concat[3-i])
    else: 
        blended = cv2.add(blended,concat[3-i])
cv2.imwrite('blended.png',blended)