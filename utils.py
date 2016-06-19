from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import numpy as np
import random
import matplotlib.pyplot as plt


def get_datasets():

    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    target = mnist['target']

    data = preprocessing.scale(data)

    train_ordered = data[:60000]
    train_labels_ordered = target[:60000]
    training_data = zip(train_ordered, train_labels_ordered)
    random.shuffle(training_data)
    train = np.array([p[0] for p in training_data])
    train_labels = np.array([p[1] for p in training_data])

    train_outs = np.array([output_list(i, 10)
                           for i in train_labels])
    test = data[60001:]
    test_labels = target[60001:]
    test_outs = np.array([output_list(i, 10)
                          for i in test_labels])
    return train, train_outs, test, test_outs


def output_list(index, size):
    array = [0 for i in xrange(size)]
    array[int(index)] = 1
    return array


def get_confusion_matrix(corr, pred):

    matrix = np.zeros([10, 10], dtype=int)
    for i in xrange(len(corr)):
        matrix[corr[i]][pred[i]] += 1
    np.set_printoptions(suppress=True)
    return matrix


def plot_weights(W, fn):
    fig, axes = plt.subplots(10, 10)
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = W.min(), W.max()
    for coef, ax in zip(W.T, axes.ravel()):
        ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.savefig(fn)
