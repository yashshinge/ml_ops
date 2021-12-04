""""""

import sys
import argparse
import logging


def set_logger(level):
    """"""
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    logger_.addHandler(handler)
    return logger_


def get_args():
    """"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_level', default='INFO', type=str, metavar='', help='Set log level')
    args_ = vars(parser.parse_args())
    return args_
