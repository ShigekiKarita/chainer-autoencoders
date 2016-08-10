# coding: utf-8
import pickle
d = pickle.load(open("./histories.pkl", "rb"))

import pylab
for k, v in d.items():
    pylab.plot(v, label=k)

import numpy
h = numpy.array(d.values())
pylab.ylim([numpy.min(h), numpy.max(h)])
pylab.yscale("log")
pylab.legend()
pylab.draw()
pylab.show()
