# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:13:59 2019

@author: Bananin
"""

def showSaveResults (original, clustering, truth, title):
    import matplotlib.pyplot as plt
    # show the original image,
    f, axarr = plt.subplots(1, 3, figsize = (10, 40))
    axarr[0].imshow(original)
    axarr[0].axis('off')
    axarr[0].set_title("Original")
    # this clustering,
    axarr[1].imshow(clustering)
    axarr[1].axis('off')
    axarr[1].set_title("Clustering")
    # and the ground truth
    axarr[2].imshow(truth)
    axarr[2].axis('off')
    axarr[2].set_title("Truth")
    plt.subplots_adjust(wspace=0, hspace=0)
    # store and show the results
    f.savefig(title+".png", bbox_inches="tight", pad_inches=0)
    plt.show()