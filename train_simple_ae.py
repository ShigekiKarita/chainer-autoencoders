#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""

from __future__ import print_function

import numpy
import chainer
import six
from chainer import optimizers


def learning_loop(xp, dataset, args, model, optimizer):
    # Prepare dataset
    N = 60000
    train, test = dataset
    x_train, y_train = train._datasets
    x_test, y_test = test._datasets
    N_test = y_test.size

    for epoch in six.moves.range(1, n_epoch + 1):
        print('epoch', epoch)

        # training
        perm = numpy.random.permutation(N)
        sum_loss = 0       # total loss
        sum_rec_loss = 0   # reconstruction loss
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
            optimizer.update(model.get_loss_func(), x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)

        print('train mean loss={}, mean reconstruction loss={}'
              .format(sum_loss / N, sum_rec_loss / N))

        # evaluation
        sum_loss = 0
        sum_rec_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                 volatile='on')
            loss_func = model.get_loss_func(k=10, train=False)
            loss_func(x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)
            del model.loss

        print('test  mean loss={}, mean reconstruction loss={}'
              .format(sum_loss / N_test, sum_rec_loss / N_test))


if __name__ == '__main__':
    import net
    import cupy.random
    from utils import arguments, figures, serialization

    numpy.random.seed(0)
    cupy.random.seed(0)

    args = arguments.load_args()
    batchsize = args.batchsize
    n_epoch = args.epoch
    n_latent = args.dimz

    # Prepare VAE model, defined in net.py
    # model = net.VAE(784, n_latent, 500)
    # model = net.DeepAutoEncoder(784, n_latent)
    model = net.SparseAutoEncoder(784, n_latent)
    xp = arguments.set_device(args, model)

    # Setup optimizer
    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)

    dataset = chainer.datasets.get_mnist()
    t = (args, model, optimizer)
    serialization.load(*t)
    learning_loop(xp, dataset, *t)
    serialization.save(*t)
    figures.execute(model, dataset, n_latent)
