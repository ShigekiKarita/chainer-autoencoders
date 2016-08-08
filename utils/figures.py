import chainer
import matplotlib.pyplot as plt
import numpy as np

# original images and reconstructed images
def save_images(x, filename):
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        ai.imshow(xi.reshape(28, 28)) # interpolation="none")
    fig.savefig("res/" + filename)


def execute(model, dataset, n_latent):
    train, test = dataset
    x_train, y_train = train._datasets
    x_test, y_test = test._datasets

    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(x_train[train_ind]), volatile='on')
    x1 = model(x)
    save_images(x.data, 'train')
    save_images(x1.data, 'train_reconstructed')

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(x_test[test_ind]), volatile='on')
    x1 = model(x)
    save_images(x.data, 'test')
    save_images(x1.data, 'test_reconstructed')

    # draw images from randomly sampled z
    z = chainer.Variable(np.random.normal(0, 1, (9, n_latent)).astype(np.float32))
    x = model.decode(z)
    save_images(x.data, 'sampled')
