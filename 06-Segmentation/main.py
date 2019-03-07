#!/home/afromero/anaconda3/bin/ipython

def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt=sio.loadmat(img_file.replace('jpg', 'mat'))
    segm=gt['groundTruth'][0,5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
    
def check_dataset(folder):
    import os
    if not os.path.isdir(folder):
        # download, unzip and remove .zip
        os.system("wget http://157.253.196.67/BSDS_small.zip")
        os.system("unzip BSDS_small.zip")
        os.system("rm BSDS_small.zip")

if __name__ == '__main__':
    import argparse
    from watershed import watershed
    import numpy as np
    from random import randint
    import matplotlib.pyplot as plt
    from skimage import io, color
    from sklearn.metrics import mutual_info_score
    import scipy.io as sio
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', type=str, default='rgb', choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']) # If you use more please add them to this list.
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)
    opts = parser.parse_args()
    
    # check for dataset, download it if necessary
    check_dataset(opts.img_file.split('/')[0])
    # read the image and make necessary transformations
    img = io.imread("BSDS_small/train/"+opts.img_file)
    
    # map to the correct color space
    if "lab" in opts.color:
        img = color.rgb2lab(img)
    elif "hsv" in opts.color:
        img = color.rgb2hsv(img)
    # add spatial dimensions if requested. Doesn't make sense for watershed clustering
    if "xy" in opts.color and opts.method!="watershed":
        temp = img
        img = np.zeros((img.shape[0],img.shape[1],img.shape[2]+2))
        img[:,:,0:temp.shape[2]] = temp
        img[:,:,temp.shape[2]] = np.array(range(img.shape[0])).reshape(img.shape[0],1)
        img[:,:,temp.shape[2]+1] = np.array(range(img.shape[1])).reshape(1,img.shape[1])
    
    # execute the requested clustering method
    if "watershed" in opts.method:
        clustering = watershed(img, opts.k)
    if "kmeans" in opts.method:
        clustering = kmeans(img, opts.k)
    if "gmm" in opts.method:
        clustering = gmm(img, opts.k)
    if "hierarchical" in opts.method:
        clustering = hierarchical(img, opts.k)
        
    # show the original image,
    f, axarr = plt.subplots(1, 3, figsize = (10, 40))
    axarr[0].imshow(img)
    axarr[0].axis('off')
    axarr[0].set_title("Original")
    # this clustering,
    axarr[1].imshow(clustering)
    axarr[1].axis('off')
    axarr[1].set_title("Clustering")
    # and the ground truth
    truth = sio.loadmat("BSDS_small/train/"+opts.img_file.replace('jpg', 'mat'))
    segm = truth['groundTruth'][0,4][0][0]['Segmentation']
    axarr[2].imshow(segm)
    axarr[2].axis('off')
    axarr[2].set_title("Truth")
    # a nice title
    result_title = opts.img_file+"_k="+str(opts.k)+"_"+opts.method
    plt.subplots_adjust(wspace=0, hspace=0)
    # store the results
    f.savefig(result_title.replace(".jpg","")+".png", bbox_inches="tight", pad_inches=0)
    # f.show()
    
    # calculate and report mutual information
    print("Mutual information: "+str(mutual_info_score(segm.flatten(),clustering.flatten())))
    print("Mutual information between truth and uniformly random k-clustering: "+str(mutual_info_score(segm.flatten(),[randint(0,opts.k) for i in range(segm.flatten().shape[0])])))