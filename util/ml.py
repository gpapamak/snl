from itertools import izip

import numpy as np
import theano
import theano.tensor as tt


def select_theano_act_function(name, dtype=theano.config.floatX):
    """
    Given the name of an activation function, returns a handle for the corresponding function in theano.
    """

    if name == 'logistic':
        clip = 15.0 if dtype == 'float32' else 19.0
        f = lambda x: tt.nnet.sigmoid(tt.clip(x, -clip, clip))

    elif name == 'tanh':
        clip = 9.0 if dtype == 'float32' else 19.0
        f = lambda x: tt.tanh(tt.clip(x, -clip, clip))

    elif name == 'linear':
        f = lambda x: x

    elif name == 'relu':
        f = tt.nnet.relu

    elif name == 'softplus':
        f = tt.nnet.softplus

    elif name == 'softmax':
        f = tt.nnet.softmax

    else:
        raise ValueError(name + ' is not a supported activation function type.')

    return f


def copy_model_parms(source_model, target_model):
    """
    Copies the parameters of source_model to target_model.
    """

    for sp, tp in izip(source_model.parms, target_model.parms):
        tp.set_value(sp.get_value())


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[xrange(labels.size), labels] = 1

    return y


def are_parms_finite(model):
    """
    Check whether all parameters of a model are finite.
    :param model: an ml model
    :return: False if at least one parameter is inf or nan
    """

    check = True

    for p in model.parms:
        check = check and np.all(np.isfinite(p.get_value()))

    return check


def total_norm_constraint(xs, max_norm):
    """
    Rescales a list of theano tensors xs to have total norm no more than max_norm.
    Adapted from lasagne.updates.
    :param xs: list of theano tensors
    :max_norm: maximum total norm
    """

    cast = np.dtype(theano.config.floatX).type
    max_norm = cast(max_norm)
    eps = cast(1.0e-7)

    norm = tt.sqrt(sum(tt.sum(x**2) for x in xs))
    target_norm = tt.clip(norm, 0.0, max_norm)
    multiplier = target_norm / (eps + norm)
    xs_scaled = [x * multiplier for x in xs]

    return xs_scaled
