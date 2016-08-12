# coding: utf-8
from os import path
from glob import glob
import pickle
import pylab
import numpy


def plot_loss():
    d = pickle.load(open("res/histories.pkl", "rb"))
    for k, v in d.items():
        if k.startswith("variational"):
            continue
        l = "--" if k.endswith("train") else "--"
        pylab.plot(v[:21], label=k, linestyle=l)

    h = numpy.array(d.values())
    pylab.ylim([numpy.min(h), numpy.max(h)])
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
    writer = imageio.get_writer(path + "/test.gif", fps=4)
    for i in range(0, len(imgs), 5):
        img = imresize(imgs[i], size=0.5)
        writer.append_data(img)
    writer.close()



if __name__ == '__main__':
    # plot_gif("res__/deep")
    plot_loss()
    for p in glob("res/*"):
        if path.isdir(p):
            print("plot:" + p)
            plot_gif(p)
