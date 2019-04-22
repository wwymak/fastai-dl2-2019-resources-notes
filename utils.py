from pathlib import Path
from fastai import datasets
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
    DATA_DIR = Path("/Users/wwymak/computer-science-courses/kmnist")
    X_train = np.load(DATA_DIR / "kmnist-train-imgs.npz")['arr_0'].astype('float')
    X_test = np.load(DATA_DIR / "kmnist-test-imgs.npz")['arr_0'].astype('float')
    y_train = np.load(DATA_DIR / "kmnist-train-labels.npz")['arr_0'] #.astype('float')
    y_test = np.load(DATA_DIR / "kmnist-test-labels.npz")['arr_0'] #.astype('float')

    train_mean = X_train.mean()
    train_std = X_train.std()

    X_train = normalise(X_train, train_mean, train_std)
    X_test = normalise(X_test, train_mean, train_std)

    X_train, X_test, y_train, y_test = map(from_numpy, (X_train, X_test, y_train, y_test))
    X_train, X_test = [x.float() for x in (X_train, X_test)]
    y_train, y_test = [x.long() for x in (y_train, y_test)]

    X_train = X_train.reshape([X_train.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, X_test, y_train, y_test