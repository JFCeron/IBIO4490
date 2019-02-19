# manipulacion de imagenes
from PIL import Image
# matematicas
import numpy as np   
# imagenes para mezclar
dante = Image.open("dante.jpg")
pato = Image.open("whiteduck.jpg")
imgs = [dante,pato]
# ajustar tamanos
tamano = (np.mean([dante.size[0],pato.size[0]]),np.mean([dante.size[1],pato.size[1]]))
