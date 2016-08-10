#!/usr/bin/env python
from __future__ import print_function

import numpy
import chainer
from chainer import optimizers


def learning_loop(name, xp, dataset, args, model, optimizer):
    # Prepare dataset
    N = 60000
    train, test = dataset
    x_train, y_train = train._datasets
    x_test, y_test = test._datasets
    N_test = y_test.size
    batchsize = args.batchsize
    train_history = []
    test_history = []

    for epoch in range(1, args.epoch + 1):
        print('epoch', epoch)
        fmt = name + "/%s_" + "%05d" % epoch

        # training
        perm = numpy.random.permutation(N)
        sum_loss = 0       # total loss
        sum_rec_loss = 0   # reconstruction loss
        for i in range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
            optimizer.update(model.get_loss_func(), x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)

        print('train mean loss={}, mean reconstruction loss={}'
              .format(sum_loss / N, sum_rec_loss / N))
        train_history.append(sum_rec_loss / N)

        # evaluation
        sum_loss = 0
        sum_rec_loss = 0
        for i in range(0, N_test, batchsize):
            x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                 volatile='on')
            loss_func = model.get_loss_func(k=10, train=False)
            loss_func(x)
            sum_loss += float(model.loss.data) * len(x.data)
            sum_rec_loss += float(model.rec_loss.data) * len(x.data)
            del model.loss

        print('test  mean loss={}, mean reconstruction loss={}'
              .format(sum_loss / N_test, sum_rec_loss / N_test))
        test_history.append(sum_rec_loss / N_test)

        figures.execute(fmt, model.to_cpu(), dataset)
        arguments.set_device(args, model)

    return train_history, test_history

if __name__ == '__main__':
    import net
    import cupy.random
    from utils import arguments, figures, serialization

    numpy.random.seed(0)
    cupy.random.seed(0)

    args = arguments.load_args()
    dataset = chainer.datasets.get_mnist()
    n_latent = args.dimz
    n_input = 784

    models = {
        "simple": net.SimpleAutoEncoder(n_input, n_latent),
        "sparse": net.SparseAutoEncoder(n_input, n_latent),
        "deep": net.DeepAutoEncoder(n_input, n_latent, n_depth=4),
        "convolutional": net.ConvolutionalAutoEncoder(n_input),
        "variational": net.VariationalAutoEncoder(784, n_latent, n_h=500)
    }

    histories = dict()
    for name, model in models.items():
        print("optimizing: %s autoencoder..." % name)
        xp = arguments.set_device(args, model)
        optimizer = optimizers.AdaDelta()
        optimizer.setup(model)

        t = (args, model, optimizer)
        serialization.load(*t)
        train, test = learning_loop(name, xp, dataset, *t)
        histories[name + "_train"] = train
        histories[name + "_test"] = test
        serialization.save(name, *t)
        # figures.execute(name + "", model, dataset)
    import pickle
    with open("histories.pkl", 'wb') as f:
        pickle.dump(histories, f)
    figures.plot_loss(histories)