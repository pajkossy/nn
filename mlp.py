import logging
import numpy as np
from argparse import ArgumentParser

from utils import get_datasets, get_confusion_matrix


def nonlin(x, deriv=False):
    if(deriv is True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

def softmax(x):
    sm =  np.exp(x)/np.sum(np.exp(x), axis=1).reshape([x.shape[0], 1])
    return sm

class MLP(object):

    def __init__(self, input_, hidden, output):
        np.random.seed(1)
        self.output = output
        self.syn0 = 2*np.random.random((input_, hidden)) - 1
        self.syn1 = 2*np.random.random((hidden, output)) - 1
        self.bias0 = (2*np.random.random((hidden, 1)) - 1).T
        self.bias1 = (2*np.random.random((output, 1)) - 1).T


    def generate_minibatches(self, X_all, y_all, it, b_size):
        for i in xrange(it):
            for j in range(X_all.shape[0]/b_size):
                yield X_all[j*b_size:(j+1)*b_size],\
                        y_all[j*b_size:(j+1)*b_size]

    def feedforward(self, X):
        l0 = X
        l1 = nonlin(np.dot(l0, self.syn0) + self.bias0)
        if self.use_crossval_error:
            l2 = softmax(np.dot(l1, self.syn1) + self.bias1)
        else:
            l2 = nonlin(np.dot(l1, self.syn1) + self.bias1)
        return l1, l2

    def backpropagate(self, l1, l2, X, y):

        if self.use_crossval_error:
            l2_delta = y - l2
        else:
            l2_error = y - l2
            l2_delta = l2_error * nonlin(l2, deriv=True)

        l1_error = l2_delta.dot(self.syn1.T)
        l1_delta = l1_error * nonlin(l1, deriv=True)

        grad1 = l1.T.dot(l2_delta)
        grad0 = X.T.dot(l1_delta)
        gradb1 = np.sum(l2_delta, axis=0)
        gradb0 = np.sum(l1_delta, axis=0)
        return gradb1, gradb0, grad1, grad0

    def train(self, X_all, y_all, it, b_size, learning_rate,
              use_crossval_error, regularization):
        
        self.regularization = regularization
        self.use_crossval_error = use_crossval_error
        j = 0
        for X, y in self.generate_minibatches(X_all, y_all, it, b_size):
            actual_size_ratio = 1.0/X.shape[0]
            j += b_size
            # feedforward
            l1, l2 = self.feedforward(X)

            # backpropagation
            gradb1, gradb0, grad1, grad0 = self.backpropagate(l1, l2, X, y)

            #regularization 
            grad1 -= self.regularization * self.syn1
            grad0 -= self.regularization * self.syn0

            self.syn1 += learning_rate * actual_size_ratio * grad1
            self.syn0 += learning_rate * actual_size_ratio * grad0
            self.bias1 += learning_rate * actual_size_ratio * gradb1
            self.bias0 += learning_rate * actual_size_ratio * gradb0

            if j > X_all.shape[0]:
                self.report_accuracy(y, l2)

                j = 0

    def report_accuracy(self, y, l2, test=False):
        corr = np.argmax(l2, axis=1)
        pred = np.argmax(y, axis=1)
        ratio = float(sum(corr == pred))/y.shape[0]
        confusion_matrix = get_confusion_matrix(corr, pred)
        if test:
            data = 'train'
        else:
            data = 'test'
        logging.info("Correctly classified: {} % on {}".format(ratio * 100, data))
        logging.info("Confusion matrix:\n{}".format(confusion_matrix))

    def evaluate(self, test, test_outs):
        _, l2 = self.feedforward(test)
        self.report_accuracy(test_outs, l2, test=True)


def read_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--hidden', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-c', '--use_crossval_error', action='store_true')
    parser.add_argument('-r', '--regularization', type=float, default=0.01)
    return parser.parse_args()


def main():

    args = read_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    network = MLP(784, args.hidden, 10)
    train, train_outs, test, test_outs = get_datasets(args.normalize)
    network.train(train,
                  train_outs,
                  args.iterations,
                  args.batch_size,
                  args.learning_rate,
                  args.use_crossval_error,
                  args.regularization)
    network.evaluate(test, test_outs)

if __name__ == "__main__":
    main()
