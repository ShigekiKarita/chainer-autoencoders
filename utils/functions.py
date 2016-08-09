import chainer
import chainer.functions as F
import numpy
from chainer import cuda, functions
from chainer.utils import type_check, force_array


def l1_norm(x):
    return F.sum(abs(x))


def lp_norm(x, p):
    return F.sum((x**p))**(1./p)


def l2_norm(x):
    return lp_norm(x, 2)


class UpSampling2D(chainer.function.Function):

    def __init__(self, size):
        self.size = size

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type = in_types[0]
        type_check.expect(
            x_type.dtype == numpy.float32
            # len(x_type.shape) == 4, # mb-size, channels, w, h
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x = inputs[0]
        xs = xp.repeat(x, self.size, axis=2)
        return xp.repeat(xs, self.size, axis=3),

    def backward(self, inputs, grad_outputs):
        gap = self.size * F.AveragePooling2D(self.size).forward(grad_outputs)[0]
        return gap,


def up_sampling_2d(x, size):
    return UpSampling2D(size)(x)


class SigmoidCrossEntropy(chainer.function.Function):

    def __init__(self, use_cudnn=True, normalize=True):
        self.use_cudnn = use_cudnn
        self.normalize = normalize

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype == numpy.float32,
            x_type.shape == t_type.shape
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        # stable computation of the cross entropy.
        loss = -xp.sum((x * (t - x) - xp.log1p(xp.exp(-xp.abs(x)))))
        # loss = -xp.sum(t * xp.log(x) + (1 - t) * xp.log(1 - x))
        return loss,

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs
        gloss = grad_outputs[0]
        y, = F.Sigmoid(self.use_cudnn).forward((x,))
        gx = gloss * (y - t)
        return gx, None


def sigmoid_cross_entropy(x, t, use_cudnn=True, normalize=True):
    return SigmoidCrossEntropy(use_cudnn, normalize)(x, t)

