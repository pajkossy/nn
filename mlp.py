import logging
import numpy as np
from argparse import ArgumentParser

from utils import get_datasets, get_confusion_matrix


def logistic(x, deriv=False):
    return 1/(1+np.exp(-x))


def logistic_deriv_by_value(f_x):
    return f_x * (1-f_x)


def softmax(x):
    probs_unnormalized = np.exp(x)
    normalization = np.sum(np.exp(x), axis=1, keepdims=True)
    probs = probs_unnormalized / normalization
    return probs

def calculate_lr(orig, lr_decay_rate, t):
    # exponential decay
    return orig * np.exp(-t/lr_decay_rate) 


class MLP(object):

    def __init__(self, input_, hidden, output):
        np.random.seed(1)
        self.output = output
        self.W1 = 2 * np.random.random((input_, hidden)) - 1
        self.W2 = 2 * np.random.random((hidden, output)) - 1
        self.b1 = (2 * np.random.random((1, hidden)) - 1)
        self.b2 = (2 * np.random.random((1, output)) - 1)

    def generate_minibatches(self, X_all, y_all, it, b_size):
        for i in xrange(it):
            for j in range(X_all.shape[0]/b_size):
                yield X_all[j*b_size:(j+1)*b_size],\
                        y_all[j*b_size:(j+1)*b_size]

    def feedforward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = logistic(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        if self.use_crossval_error:
            a2 = softmax(z2)
        else:
            a2 = logistic(z2)
        return a1, a2

    def backpropagate(self, a1, a2, X, y):
        '''
        notation:
        dE/d_output of layer {index}: error_dout_{index}
        dE/d_input of layer {index}: delta_{index}
        '''
        if self.use_crossval_error:
            probs = a2
            delta_2 = probs - y
        else:
            error_dout_2 = a2 - y
            delta_2 = error_dout_2 * logistic_deriv_by_value(a2)

        error_dout_1 = delta_2.dot(self.W2.T)
        delta_1 = error_dout_1 * logistic_deriv_by_value(a1)

        dW2 = a1.T.dot(delta_2)
        dW1 = X.T.dot(delta_1)
        db2 = np.sum(delta_2, axis=0)
        db1 = np.sum(delta_1, axis=0)
        return db2, db1, dW2, dW1

    def normalize_gradients(self, db2, db1, dW2, dW1, mbatch_size):
        return (db2/mbatch_size,
                db1/mbatch_size,
                dW2/mbatch_size,
                dW1/mbatch_size)

    def train(self, X_all, y_all, it, b_size, orig_learning_rate,
              use_crossval_error, reg_lambda, lr_decay_rate):

        self.reg_lambda = reg_lambda
        self.use_crossval_error = use_crossval_error
        j = 0
        t = -1.0
        for X, y in self.generate_minibatches(X_all, y_all, it, b_size):
            t += 1
            learning_rate = calculate_lr(
                orig_learning_rate, lr_decay_rate, t)
            j += b_size
            # feedforward
            a1, a2 = self.feedforward(X)

            # backpropagation
            db2, db1, dW2, dW1 = self.backpropagate(a1, a2, X, y)

            # normalize by minibatch size
            db2, db1, dW2, dW1 = self.normalize_gradients(
                db2, db1, dW2, dW1, X.shape[0])

            # add regularization term to weight updates
            dW2 += self.reg_lambda * self.W2
            dW1 += self.reg_lambda * self.W1

            self.W2 -= learning_rate * dW2
            self.W1 -= learning_rate * dW1
            self.b2 -= learning_rate * db2
            self.b1 -= learning_rate * db1

            if j > X_all.shape[0]:
                self.report_accuracy(y, a2)
                logging.info('Learning rate: {}'.format(learning_rate))

                j = 0

    def report_accuracy(self, y, a2, test=False):
        corr = np.argmax(a2, axis=1)
        pred = np.argmax(y, axis=1)
        ratio = float(sum(corr == pred))/y.shape[0]
        confusion_matrix = get_confusion_matrix(corr, pred)
        if test:
            data = 'test'
        else:
            data = 'train'
        logging.info("Correctly classified: {} % on {}".format(
            ratio * 100, data))
        logging.info("Confusion matrix:\n{}".format(confusion_matrix))

    def evaluate(self, test, test_outs):
        _, a2 = self.feedforward(test)
        self.report_accuracy(test_outs, a2, test=True)


def read_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--hidden', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-i', '--iterations', type=int)
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-d', '--lr_decay_rate', type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-c', '--use_crossval_error', action='store_true')
    parser.add_argument('-r', '--reg_lambda', type=float, default=0.01)
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
                  args.reg_lambda, 
                  args.lr_decay_rate)
    network.evaluate(test, test_outs)

if __name__ == "__main__":
    main()
