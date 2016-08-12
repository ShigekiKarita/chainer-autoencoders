import os
from os import path

import chainer
import matplotlib.pyplot as plt
import numpy as np
import pylab


# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28)) # interpolation="none")

    fig.suptitle(filename, fontsize="x-large")

    filename = "res/" + filename
    d = path.dirname(filename)
    if not path.exists(d):
        os.makedirs(d)
    fig.savefig(filename)
    plt.clf()
    plt.close('all')



def execute(fmt, model, dataset):
    train, test = dataset
    x_train, y_train = train._datasets
    x_test, y_test = test._datasets

    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(x_train[train_ind]), volatile='on')
    x1 = model(x)
    # save_images(x.data, fmt % 'train')
    save_images(x1.data, fmt % 'train_reconstructed')

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(x_test[test_ind]), volatile='on')
    x1 = model(x)
    # save_images(x.data, fmt % 'test')
    save_images(x1.data, fmt % 'test_reconstructed')

    try:
        # draw images from randomly sampled z
        r = np.random.normal(0, 1, (9, model.n_latent))
        z = chainer.Variable(r.astype(np.float32))
        x = model.decode(z)
        save_images(x.data, fmt % 'sampled')
    except AttributeError:
        print("skip sampling because %s has no self.n_latent" % str(model))


import numpy
def plot_loss(histories):
    for k, v in histories.items():
        pylab.plot(v, label=k)

    h = numpy.array(histories.values())
    pylab.ylim([numpy.min(h), numpy.max(h)])
    pylab.yscale("log")
    pylab.legend()
    pylab.draw()
    pylab.savefig("loss.pdf")
    pylab.show()