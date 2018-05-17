import cPickle as pickle
import os
import sys


def save(data, file):
    """
    Saves data to a file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file + '.pkl', 'w') as f:
        pickle.dump(data, f)


def load(file):
    """
    Loads data from file.
    """

    with open(file + '.pkl', 'r') as f:
        data = pickle.load(f)

    if hasattr(data, 'reset_theano_functions'):
        data.reset_theano_functions()

    return data


def save_txt(str, file):
    """
    Saves string to a text file.
    """

    dir = os.path.dirname(file)
    if dir:
        make_folder(dir)

    with open(file, 'w') as f:
        f.write(str)


def load_txt(file):
    """
    Loads string from text file.
    """

    with open(file, 'r') as f:
        str = f.read()

    return str


def make_folder(folder):
    """
    Creates given folder (or path) if it doesn't exist.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)


class Logger:
    """
    Implements an object that logs messages to a file, as well as printing them on the sceen.
    """

    def __init__(self, filename):
        """
        :param filename: file to be created for logging
        """
        self.f = open(filename, 'w')

    def write(self, msg):
        """
        :param msg: string to be logged and printed on screen
        """
        sys.stdout.write(msg)
        self.f.write(msg)

    def __enter__(self):
        """
        Context management enter function.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context management exit function. Closes the file.
        """
        self.f.close()
        return False
