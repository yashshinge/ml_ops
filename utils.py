""""""

import sys
import argparse
import logging

import matplotlib.pyplot as plt


def set_logger(level):
    """"""
    logger = logging.getLogger('simple_classifier')
    logger.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    return logger


def get_args():
    """"""
    parser = argparse.ArgumentParser(description="Simple classifier training job.")
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help="Set log level (default: 'INFO')")
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help="Number of epochs to train (default: 2)")
    parser.add_argument('--seed', type=int, default=918, metavar='S',
                        help="Random seed (default: 918)")
    args = parser.parse_args()
    return args


def plot_helper(x, y, plt_name='viz.png'):
    """"""
    plt.style.use('seaborn-darkgrid')
    ax = plt.axes()
    ax.set(xticks=x, xlabel="No. of epochs", ylabel='Accuracy (%)', title='Test accuracy v/s Epochs')
    _ = plt.plot(x, y)
    plt.savefig(plt_name)
