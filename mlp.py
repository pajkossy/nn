import logging
import numpy as np
from argparse import ArgumentParser

from utils import get_datasets, get_confusion_matrix, plot_weights


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
    if lr_decay_rate == 0:
        return orig
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
        a2 = self.nonlin(z2)
        return a1, a2

    def backpropagate(self, a1, a2, X, y):
        '''
        notation:
        dE/d_output of layer {index}: error_dout_{index}
        dE/d_input of layer {index}: delta_{index}
        '''
        delta_2 = self.get_delta(a2, y)
        error_dout_1 = delta_2.dot(self.W2.T)
        delta_1 = error_dout_1 * logistic_deriv_by_value(a1)

        dW2 = a1.T.dot(delta_2)
        dW1 = X.T.dot(delta_1)
        db2 = np.sum(delta_2, axis=0)
        db1 = np.sum(delta_1, axis=0)
        return db2, db1, dW2, dW1

    def get_softmax_delta(self, a2, y):
        return a2 - y

    def get_mse_delta(self, a2, y):
        error_dout_2 = a2 - y
        return error_dout_2 * logistic_deriv_by_value(a2)

    def normalize_gradients(self, db2, db1, dW2, dW1, mbatch_size):
        return (db2/mbatch_size,
                db1/mbatch_size,
                dW2/mbatch_size,
                dW1/mbatch_size)

    def train(self, X_all, y_all, it, b_size, orig_learning_rate,
              reg_lambda, lr_decay_rate, cost):

        self.reg_lambda = reg_lambda
        if cost == 'softmax':
            self.get_delta = self.get_softmax_delta
            self.nonlin = softmax
        elif cost == 'mse':
            self.get_delta = self.get_mse_delta
            self.nonlin = logistic
        epoch = 0
        t = -1.0
        for X, y in self.generate_minibatches(X_all, y_all, it, b_size):
            t += 1
            learning_rate = calculate_lr(
                orig_learning_rate, lr_decay_rate, t)
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

            if (t + 1) * b_size % X_all.shape[0] == 0:
                epoch += 1
                logging.info('Epoch {} done, learning rate: {}'\
                             .format(epoch, learning_rate))


    def report_accuracy(self, y, a2, test=False):
        corr = np.argmax(a2, axis=1)
        pred = np.argmax(y, axis=1)
        ratio = float(sum(corr == pred))/y.shape[0]
        confusion_matrix = get_confusion_matrix(corr, pred)
        logging.info("Correctly classified: {}%".format(
            ratio * 100))
        logging.info("Confusion matrix:\n{}".format(confusion_matrix))


    def evaluate(self, test, test_outs):
        _, a2 = self.feedforward(test)
        self.report_accuracy(test_outs, a2, test=True)


def read_args():
    parser = ArgumentParser()
    parser.add_argument('-s', '--hidden', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=20)
    parser.add_argument('-i', '--iterations', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.1)
    parser.add_argument('-d', '--lr_decay_rate', type=float, default=0)
    parser.add_argument('-r', '--reg_lambda', type=float, default=0.0005)
    parser.add_argument('-c', '--cost', choices=['softmax', 'mse'],
                        default='softmax')
    parser.add_argument('-p', '--plot_weights_fn', default='None')
    return parser.parse_args()


def main():

    args = read_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s : " +
        "%(module)s (%(lineno)s) - %(levelname)s - %(message)s")

    network = MLP(784, args.hidden, 10)
    train, train_outs, test, test_outs = get_datasets()
    network.train(train,
                  train_outs,
                  args.iterations,
                  args.batch_size,
                  args.learning_rate,
                  args.reg_lambda,
                  args.lr_decay_rate,
                  args.cost)
    network.evaluate(test, test_outs)
    if args.plot_weights_fn:
        plot_weights(network.W1, args.plot_weights_fn)
        logging.info('Weights plotted to {}'.format(args.plot_weights_fn))

if __name__ == "__main__":
    main()
