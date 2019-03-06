"""
Metodo que segmenta los pixeles de una imageny genera una visualizacion

•colorSpace : 'rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy' or 'hsv+xy'
•clusteringMethod = 'kmeans', 'gmm', 'hierarchical' or 'watershed'.
•numberOfClusters positive integer (larger than 2)
"""
from skimage import io, color
from skimage.filters import rank
from skimage.morphology import watershed, disk
# parametros de prueba
rgbImage = "BSDS_small/train/2092.jpg"
colorSpace = "rgb"
clusteringMethod = "watershed"
img = io.imread(rgbImage)

# mapear imagen al espacio de colores solicitado
if "lab" in colorSpace:
    img = color.rgb2lab(img)
elif "hsv" in colorSpace:
    img = color.rgb2hsv(img)
else:
    assert "rgb" in colorSpace

# ejecutar el clustering solicitado
if clusteringMethod == "kmeans":
    1+1
elif clusteringMethod == "gmm":
    1+1
elif clusteringMethod == "hierchical":
    1+1
elif clusteringMethod == "watershed":
    # remover ruido
    for i in range(img.shape[-1]):
        img[...,i] = rank.median(img[...,i], disk(2))

#def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
#    ...
#    return segmentation