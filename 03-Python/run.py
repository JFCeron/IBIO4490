# manipulacion de imagenes
from PIL import Image                                                                                
from PIL import ImageDraw
# lectura de anotaciones
from scipy.io import loadmat
# expresiones regulares
import re
# listas y matematicas
import numpy as np
# funcionalidades de carpetas
import os
# descarga
import urllib
# descompresion
import tarfile
# tiempo de ejecucion
import time

#comenzamos a contar
start = time.time()

# DESGARGA Y DESCOMPRESION
tarname = "HomeObjects.tar"
folder = "HomeObjects"
# descargar .tar si aun no se ha hecho
if not os.path.isdir(folder):
    urllib.request.urlretrieve("http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/HomeObjects06.tar",tarname)
    tarobj = tarfile.open(tarname)
    tarobj.extractall(folder)
    tarobj.close()
# VISUALIZACION DE ANOTACIONES
# tamano de los thumbnails
size = 256,256
# carpeta donde viven las imagenes
imgfolder = folder+'/Train/'
# obtener muestra aleatoria de 9 fotos
train = np.array(os.listdir(imgfolder))
images = train[[bool(re.search("JPG",train[i])) for i in range(len(train))]]
subsample = np.random.choice(images, 9, False)
# marco para la grilla de imagenes
background = Image.new('RGBA',(256*3, 256*3), (255, 255, 255, 255))
for i in range(len(subsample)):
    imgfile = imgfolder+subsample[i]
    img = Image.open(imgfile)
    anotacion = loadmat(imgfolder+'Gtruth/'+subsample[i]+'.mat')
    points = anotacion['outline'][0,0][0].flatten().tolist()
    # para que la figura cierre
    points = points+points[0:2]
    # graficar anotacion
    draw = ImageDraw.Draw(img)
    draw.line(points, fill='cyan', width=6)
    # resize
    img = img.resize(size)
    del draw
    # agregar imagen al marco
    background.paste(img, ((i%3)*256+1,int((i-i%3)*256/3+1)))
#resultado    
background.show()

print("Tiempo de ejecucion: "+str(time.time()-start)+" segundos")