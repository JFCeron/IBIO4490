# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:15:43 2019

@author: secg9
"""
import os
from segmentarImagen import clusterImages
from pathlib import Path
import pdb
import time    
imagenes = os.listdir('./test/')

imagenes = [ y for y in imagenes if y.split('.')[-1] == 'jpg']



os.mkdir('watershed23')
start_time=time.time()
[clusterImages(y,"watershed") for y in imagenes]
print("lololo" % (time.time()- start_time))
