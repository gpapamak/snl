import numpy as np
import theano
import theano.tensor as tt


def SquareError(x):
    """Square error loss function."""

    if x.ndim == 1:
        y = tt.vector('y')
        L = tt.mean((x - y) ** 2)

    elif x.ndim == 2:
        y = tt.matrix('y')
        L = tt.mean(tt.sum((x - y) ** 2, axis=1))

    else:
        raise ValueError('x must be either a vector or a matrix.')

    L.name = 'loss'

    return y, L


def CrossEntropy(x):
    """Cross entropy loss function. Only works for networks with one output."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x[:, 0]

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = tt.vector('y')
    L = -tt.mean(y * tt.log(x) + (1-y) * tt.log(1-x))
    L.name = 'loss'

    return y, L


def MultiCrossEntropy(x):
    """Cross entropy loss function with multiple outputs."""

    assert x.ndim == 2, 'x must be a matrix.'

    y = tt.matrix('y')
    L = -tt.mean(tt.sum(y * tt.log(x), axis=1))
    L.name = 'loss'

    return y, L


def Accuracy(x):
    """Accuracy loss function. Mainly useful for validation."""

    if x.ndim == 1:
        pass

    elif x.ndim == 2:
        x = x.argmax(axis=1)

    else:
        raise ValueError('x must be either a vector or a matrix.')

    y = tt.vector('y')
    L = tt.mean(tt.eq(y, x))
    L.name = 'loss'

    return y, L


def WeightDecay(ws, wdecay):
    """Weight decay regularization."""

    assert wdecay > 0.0

    L = (wdecay / 2.0) * sum([tt.sum(w**2) for w in ws])
    return L


def SviRegularizer(mps, sps, wdecay):
    """
    Regularizer for stochastic variational inference, assuming the variational posterior is a gaussian with diagonal
    covariance and the prior is a spherical zero-centred gaussian whose precision is the weight decay parameter.
    :param mps: means of variational posterior (list of theano variables)
    :param sps: log stds of variational posterior (list of theano variables)
    :param wdecay: weight decay (real value)
    """

    assert wdecay > 0.0

    n_params = sum([mp.get_value().size for mp in mps])

    L1 = 0.5 * wdecay * (sum([tt.sum(mp**2) for mp in mps]) + sum([tt.sum(tt.exp(sp*2)) for sp in sps]))
    L2 = sum([tt.sum(sp) for sp in sps])
    Lc = 0.5 * n_params * (1.0 + np.log(wdecay))

    L = L1 - L2 - Lc

    return L


def SviRegularizer_DiagCov(mps, sps, m0s, s0s):
    """
    Regularizer for stochastic variational inference, assuming both the prior and the variational posterior are
    gaussians with diagonal covariances.
    :param mps: means of variational posterior (list of theano variables)
    :param sps: log stds of variational posterior (list of theano variables)
    :param m0s: means of prior (list of numpy arrays)
    :param s0s: log stds of prior (list of numpy arrays)
    """

    n_params = sum([mp.get_value().size for mp in mps])

    # m0s and s0s are numpy arrays, so turn them into shared theano variables
    m0s = [theano.shared(m0, borrow=True) for m0 in m0s]
    s0s = [theano.shared(s0, borrow=True) for s0 in s0s]

    L1 = 0.5 * sum([tt.sum(tt.exp(-2.0 * s0) * (mp - m0) ** 2) for mp, m0, s0 in zip(mps, m0s, s0s)])
    L2 = 0.5 * sum([tt.sum(tt.exp(2.0 * (sp - s0))) for sp, s0 in zip(sps, s0s)])
    L3 = sum([tt.sum(s0 - sp) for sp, s0 in zip(sps, s0s)])

    L = L1 + L2 + L3 - 0.5 * n_params

    return L
