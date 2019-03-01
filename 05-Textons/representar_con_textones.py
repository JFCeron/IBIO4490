import numpy as np
from assignTextons import assignTextons
from fbRun import fbRun
def representar_con_textones(cubo, N, C, textons, fb):
    assert len(cubo.shape) == 4
    cubo = cubo.reshape((N, cubo.shape[2], cubo.shape[3]))
    for i in range(cubo.shape[0]):
        cubo[i,:,:] = assignTextons(fbRun(fb,cubo[i,:,:]),textons.transpose())
    cubo = cubo.reshape(cubo.shape[0], cubo.shape[1]*cubo.shape[2])
    return cubo
