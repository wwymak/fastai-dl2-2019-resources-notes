from pathlib import Path
from torchvision import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor, from_numpy, flatten
import operator
from functools import partial

import pandas as pd
import numpy as np


def normalise(x, mean, std):
    return (x - mean) /std


def get_data():
    data_dir = Path("/media/wwymak/Storage/kmnist")
    data_dir.mkdir(exist_ok=True)
    kmnist = datasets.KMNIST(data_dir, download=True)
    kmnist_test = datasets.KMNIST(data_dir, train=False)

    X_train = kmnist.data
    y_train = kmnist.targets
    X_test = kmnist_test.data
    y_test = kmnist_test.targets

    X_train = X_train.view(-1, 28 * 28).float()
    X_test = X_test.view(-1, 28 * 28).float()

    # train_mean = X_train.mean()
    # train_std = X_train.std()
    #
    # X_train = normalise(X_train, train_mean, train_std)
    # X_test = normalise(X_test, train_mean, train_std)
    #
    # X_train, X_test, y_train, y_test = map(from_numpy, (X_train, X_test, y_train, y_test))
    # X_train, X_test = [x.float() for x in (X_train, X_test)]
    # y_train, y_test = [x.long() for x in (y_train, y_test)]
    #
    # X_train = X_train.reshape([X_train.shape[0], -1])
    # X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, X_test, y_train, y_test


def get_data_mnist():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))
