from sklearn.datasets import fetch_mldata
import numpy as np
import random
import matplotlib.pyplot as plt


def get_datasets():

    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    target = mnist['target']

    data = (data - data.mean(axis=0))
    std = data.std(axis=0)
    data[:, std > 0] /= std[std > 0]

    train_split = 60000
    output_size = 10

    train_ordered = data[:train_split]
    train_labels_ordered = target[:train_split]
    training_data = zip(train_ordered, train_labels_ordered)
    random.shuffle(training_data)
    train = np.array([p[0] for p in training_data])
    train_labels = np.array([p[1] for p in training_data])

    train_outs = np.array([one_hot(i, output_size)
                           for i in train_labels])
    test = data[train_split:]
    test_labels = target[train_split:]
    test_outs = np.array([one_hot(i, output_size)
                          for i in test_labels])
    return train, train_outs, test, test_outs


def one_hot(index, size):
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
