import argparse
from chainer import cuda
import numpy as np


def load_args():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--initmodel', '-i', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=32, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=2048,
                        help='learning minibatch size')
    parser.add_argument('--model', '-m', type=str, default="any")
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    return args


def set_device(args, model):
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    return np if args.gpu < 0 else cuda.cupy