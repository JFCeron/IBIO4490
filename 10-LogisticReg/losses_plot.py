import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# regression = "softmax" OR regression = "logistic"
def losses_plot (regression):
    loss_files = sorted(os.listdir("losses"))
    train_loss = []
    val_loss = []
    lr = []
    for loss_file in loss_files:
        if regression in loss_file:
            losses = np.load("losses/"+loss_file)
            train_loss.append(losses[0,:])
            val_loss.append(losses[1,:])
            lr.append(loss_file.split("lr")[1].split(".npy")[0])
    # transform and check integrity
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    assert train_loss.shape==val_loss.shape,  "Different dimension matrices provided."
    # forget about trailing zeros
    not_trailing_zeros = np.prod(train_loss * val_loss, axis=0) > 0
    train_loss = train_loss[:,not_trailing_zeros]
    val_loss = val_loss[:,not_trailing_zeros]

    # set color scales for train (blue) and validation (red) set
    norm = mpl.colors.Normalize(vmin=0, vmax=train_loss.shape[0])
    cmap_train = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap_train.set_array([])
    cmap_val = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
    cmap_val.set_array([])
    # one x tick per epoch
    x_ticks = range(train_loss.shape[1])
    # plot train and test losses at R learning rates across N epochs
    fig, ax = plt.subplots(dpi=100)
    for i in range(train_loss.shape[0]):
       ax.plot(x_ticks, train_loss[i,:], c=cmap_train.to_rgba(i + 1))
       ax.plot(x_ticks, val_loss[i,:], c=cmap_val.to_rgba(i + 1))
    plt.gca().legend(('Train loss, smallest lr','Val loss, smallest lr'))
    plt.savefig("losses_"+regression)
    
