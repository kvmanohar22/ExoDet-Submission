import os
import pickle as pkl
import sys

from hyperopt.pyll import scope
from hyperopt.pyll_utils import validate_label

from utils.processing_helper import make_dir_if_not_exists


class Logger(object):
    """
    Logger class to print output to terminal and to file
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def start_logging(filename):
    """Start logger, appending print output to given filename"""
    make_dir_if_not_exists(os.path.dirname(filename))
    sys.stdout = Logger(filename)


def stop_logging():
    """Stop logger and return print functionality to normal"""
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal


def print_dict(data_dict, print_format=" - {:.<30}: {}"):
    assert isinstance(data_dict, dict)
    for key, value in data_dict.iteritems():
        print print_format.format(key, value)


@validate_label
def quniform_int(label, *args, **kwargs):
    return scope.int(
        scope.hyperopt_param(label,
                             scope.quniform(*args, **kwargs)))


def read_pickle_if_exists(filename):
    """
    Returns pickle if exists otherwise returns None
    :param filename: filename that contains pickle
    """
    if os.path.exists(filename):
        return pkl.load(open(filename, 'rb'))
    else:
        return None


class prettyfloat(float):
    def __repr__(self):
        return "%0.2f" % self
