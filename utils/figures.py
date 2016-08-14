import os
from glob import glob
from os import path

import chainer
import matplotlib.pyplot as plt
import numpy
import pickle
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
    x = chainer.Variable(numpy.asarray(x_train[train_ind]), volatile='on')
    x1 = model(x)
    # save_images(x.data, fmt % 'train')
    save_images(x1.data, fmt % 'train_reconstructed')

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(numpy.asarray(x_test[test_ind]), volatile='on')
    x1 = model(x)
    # save_images(x.data, fmt % 'test')
    save_images(x1.data, fmt % 'test_reconstructed')

    try:
        # draw images from randomly sampled z
        r = numpy.random.normal(0, 1, (9, model.n_latent))
        z = chainer.Variable(r.astype(numpy.float32))
        x = model.decode(z)
        save_images(x.data, fmt % 'sampled')
    except AttributeError:
        print("skip sampling because %s has no self.n_latent" % str(model))


def plot_loss(histories):
    for k, v in histories.items():
        if k.startswith("variational"):
            continue
        l = "--" if k.endswith("train") else "-"
        pylab.plot(v[:21], label=k, linestyle=l)

    h = numpy.array(histories.values())
    # pylab.ylim([numpy.min(h), numpy.max(h)])
    pylab.yscale("log")
    pylab.legend(loc="best")

    pylab.draw()
    pylab.savefig("res/loss.png")
    # pylab.show()


def plot_gif(path):
    import imageio
    from scipy.misc import imresize

    ps = glob(path + "/test*.png")
    ps = sorted(ps)
    imgs = map(imageio.imread, ps)
    writer = imageio.get_writer(path + "/test.mp4", fps=4)
    for i in imgs:
        i = imresize(i, size=0.5)
        writer.append_data(i)
    writer.close()


def main(histories):
    # plot_gif("res__/deep")
    plot_loss(histories)
    for p in glob("res/*"):
        if path.isdir(p):
            print("plot:" + p)
            plot_gif(p)


if __name__ == '__main__':
    main(pickle.load(open("res/histories.pkl", "rb")))
