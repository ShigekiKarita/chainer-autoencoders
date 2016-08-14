import doctest

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence

from utils.functions import sigmoid_cross_entropy, up_sampling_2d


class AutoEncoderBase(chainer.Chain):

    def encode(self, x):
        pass

    def decode_bottleneck(self, z):
        pass

    def decode(self, z):
        return F.sigmoid(self.decode_bottleneck(z))

    def bottleneck(self, x):
        return self.decode_bottleneck(self.encode(x))

    def __call__(self, x):
        return self.decode(self.encode(x))

    def get_loss_func(self, *args, **kwargs):
        def lf(x):
            y = self.bottleneck(x)
            self.loss = sigmoid_cross_entropy(y, x)
            # self.rec_loss = F.mean_squared_error(F.sigmoid(y), x)
            self.rec_loss = self.loss / y.data.shape[0]
            return self.loss
        return lf


class SimpleAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in, n_units):
        super(SimpleAutoEncoder, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_in),
        )

    def encode(self, x):
        return F.relu(self.l1(x))

    def decode_bottleneck(self, z):
        return self.l2(z)

    
class SparseAutoEncoder(SimpleAutoEncoder):

    def get_loss_func(self, l1=0.01, l2=0.0, *args, **kwargs):
        from utils import functions
        def lf(x):
            y = self.bottleneck(x)
            self.loss = sigmoid_cross_entropy(y, x)
            # self.rec_loss = F.mean_squared_error(F.sigmoid(y), x)
            self.rec_loss = self.loss / y.data.shape[0]
            self.loss += functions.l1_norm(y) * l1
            return self.loss
        return lf


class DeepAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in, n_depth=1, n_units=None):
        super(DeepAutoEncoder, self).__init__()
        self.n_in = n_in
        self.n_units = n_units
        self.encode_label = "encode%d"
        self.decode_label = "decode%d"
        self.n_depth = 0
        self._init_layers(n_depth)

    def encode(self, x):
        for n in range(self.n_depth):
            x = F.relu(self[self.encode_label % n](x))
        return x

    def decode_bottleneck(self, z):
        d = self.n_depth
        for n in reversed(range(1, d)):
            z = F.relu(self[self.decode_label % n](z))
        return self[self.decode_label % 0](z)

    def _init_layers(self, n_depth):
        """
        >>> [p.data.shape for p in DeepAutoEncoder(784, n_depth=2, n_units=32).params()]
        [(64, 784), (64,), (784, 64), (784,), (32, 64), (32,), (64, 32), (64,)]
        >>> [p.data.shape for p in DeepAutoEncoder(784, n_depth=2).params()]
        [(392, 784), (392,), (784, 392), (784,), (196, 392), (196,), (392, 196), (392,)]
        """
        first = None
        if self.n_units:
            first = self.n_units * (2 ** (n_depth - 1))
        self.add_layer(first)
        for n in range(1, n_depth):
            self.add_layer()

    def encoded_size(self):
        """
        >>> DeepAutoEncoder(3, n_units=7).encoded_size()
        7
        >>> DeepAutoEncoder(9, n_depth=2).encoded_size()
        2
        """
        if self.n_depth == 0:
            return self.n_in
        last = self.n_depth - 1
        encode_last = self[self.encode_label % last]
        return encode_last.b.data.size

    def add_layer(self, n_out=None):
        i = self.encoded_size()
        o = i // 2 if n_out is None else n_out
        self.add_link(self.encode_label % self.n_depth, F.Linear(i, o))
        self.add_link(self.decode_label % self.n_depth, F.Linear(o, i))
        self.n_depth += 1


class ConvolutionalAutoEncoder(AutoEncoderBase):

    def __init__(self, n_in=784):
        self.n_in_square = int(n_in**0.5)
        p = 1
        q = 0
        super(ConvolutionalAutoEncoder, self).__init__(
            # encoder
            # input (1, 28, 28)
            conv0=L.Convolution2D(1, 16, 3, pad=p), # (28, 28) -> (14, 14)
            conv1=L.Convolution2D(16, 8, 3, pad=p), # (14, 14) -> (7, 7)
            conv2=L.Convolution2D(8, 8, 3, pad=p),  # (7, 7)   -> (4, 4)
            # decoder
            conv3=L.Convolution2D(8, 8, 3, pad=p),  # (4, 4)   -> (8, 8)
            conv4=L.Convolution2D(8, 8, 3, pad=p),  # (4, 4)   -> (8, 8)
            conv5=L.Convolution2D(8, 16, 3, pad=q), # (8, 8)   -> (16)
            conv6=L.Convolution2D(16, 1, 3, pad=p),
        )
        self.n_depth = 3
        self.label = "conv%d"

    def reshape_2d(self, x):
        return x.reshape(-1, 1, self.n_in_square, self.n_in_square)

    def encode(self, x):
        x.data = self.reshape_2d(x.data)
        # print(x.data.shape)
        for n in range(self.n_depth):
            # print("encoding conv%d" % n)
            conv = self[self.label % n]
            x = F.relu(conv(x))
            # print(x.data.shape)
            # print("encoding pool%d" % n)
            x = F.max_pooling_2d(x, 2)
            # print(x.data.shape)
        return x

    def decode_bottleneck(self, z):
        # print(z.data.shape)
        last = self.n_depth * 2
        for n in range(self.n_depth, last):
            # print("decoding conv%d" % n)
            conv = self[self.label % n]
            z = F.relu(conv(z))
            # print(z.data.shape)
            # print("decoding pool%d" % n)
            z = up_sampling_2d(z, 2)
            # print(z.data.shape)
        z = self[self.label % last](z)
        # print(z.data.shape)
        return z


class VariationalAutoEncoder(AutoEncoderBase):
    """Variational AutoEncoder"""
    def __init__(self, n_in, n_latent, n_h):
        super(VariationalAutoEncoder, self).__init__(
            # encoder
            le1=L.Linear(n_in, n_h),
            le2_mu=L.Linear(n_h, n_latent),
            le2_ln_var=L.Linear(n_h, n_latent),
            # decoder
            ld1=L.Linear(n_latent, n_h),
            ld2=L.Linear(n_h, n_in),
        )
        self.loss = None
        self.n_latent = n_latent

    def __call__(self, x, sigmoid=True):
        """AutoEncoder"""
        return self.decode(self.encode(x)[0], sigmoid)

    def encode(self, x):
        h1 = F.tanh(self.le1(x))
        mu = self.le2_mu(h1)
        ln_var = self.le2_ln_var(h1)  # log(sigma**2)
        return mu, ln_var

    def decode(self, z, sigmoid=True):
        h1 = F.tanh(self.ld1(z))
        h2 = self.ld2(h1)
        if sigmoid:
            return F.sigmoid(h2)
        else:
            return h2

    def get_loss_func(self, C=1.0, k=1, train=True):
        """Get loss function of VAE.

        The loss value is equal to ELBO (Evidence Lower Bound)
        multiplied by -1.

        Args:
            C (int): Usually this is 1.0. Can be changed to control the
                second term of ELBO bound, which works as regularization.
            k (int): Number of Monte Carlo samples used in encoded vector.
            train (bool): If true loss_function is used for training.
        """
        def lf(x):
            mu, ln_var = self.encode(x)
            batchsize = len(mu.data)
            # reconstruction loss
            rec_loss = 0
            for l in range(k):
                z = F.gaussian(mu, ln_var)
                rec_loss += F.bernoulli_nll(x, self.decode(z, sigmoid=False)) \
                    / (k * batchsize)
            self.rec_loss = rec_loss
            self.loss = self.rec_loss + \
                C * gaussian_kl_divergence(mu, ln_var) / batchsize
            return self.loss
        return lf


if __name__ == '__main__':
    doctest.testmod()
