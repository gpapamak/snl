from gaussian import Gaussian, fit_gaussian
from mog import MoG, fit_mog
from uniform import Uniform, BoxUniform

import numpy as np


class TruncatedPdf:
    """
    Implements a pdf which is the truncation of a given pdf to a region specified by an indicator function.
    """

    def __init__(self, pdf, indicator):
        """
        :param pdf: a pdf object
        :param indicator: function returning whether input is in support
        """

        self.pdf = pdf
        self.indicator = indicator
        self.n_dims = pdf.n_dims

    def eval(self, xs, log=True):
        """
        Evaluates the truncated pdf.
        NOTE: if in support, returns an unnormalized density.
        """

        ls = self.pdf.eval(xs, log=log)
        zero = -float('inf') if log else 0.0

        return np.where(self.indicator(xs), ls, zero)

    def gen(self, n_samples=None, rng=np.random):
        """
        Generates a number of samples using rejection sampling.
        """

        if n_samples is None:

            while True:
                x = self.pdf.gen(rng=rng)
                if self.indicator(x):
                    return x

        elif n_samples == 0:

            return self.pdf.gen(0, rng=rng)

        else:

            xs = None
            n_sofar = 0

            while True:

                xs = self.pdf.gen(n_samples, rng=rng)
                xs = xs[self.indicator(xs)]
                n_sofar = xs.shape[0]

                # break only when we have at least one sample
                if n_sofar > 0:
                    break

            n_rem = n_samples - n_sofar
            assert n_rem < n_samples

            if n_rem > 0:
                # request remaining samples
                xs_rem = self.gen(n_rem, rng=rng)
                xs = np.concatenate([xs, xs_rem], axis=0)

            assert xs.shape[0] == n_samples
            return xs


def gaussian_kde(xs, ws=None, std=None):
    """
    Returns a mixture of gaussians representing a kernel density estimate.
    :param xs: rows are datapoints
    :param ws: weights, optional
    :param std: the std of the kernel, if None then a default is used
    :return: a MoG object
    """

    xs = np.array(xs)
    assert xs.ndim == 2, 'wrong shape'

    n_data, n_dims = xs.shape
    ws = np.full(n_data, 1.0 / n_data) if ws is None else np.asarray(ws)
    var = n_data ** (-2.0 / (n_dims + 4)) if std is None else std ** 2

    return MoG(a=ws, ms=xs, Ss=[var * np.eye(n_dims) for _ in xrange(n_data)])


def test_gaussian_kde():

    pdf = MoG(a=[0.3, 0.7], ms=[[-1.0, 0.0], [1.0, 0.0]], Ss=[np.diag([0.1, 1.1]), np.diag([1.1, 0.1])])
    xs = pdf.gen(5000)
    kde = gaussian_kde(xs)

    import util.plot

    lims = [-4.0, 4.0]
    util.plot.plot_pdf_marginals(pdf, lims=lims).suptitle('original')
    util.plot.plot_hist_marginals(xs, lims=lims).suptitle('samples')
    util.plot.plot_pdf_marginals(kde, lims=lims).suptitle('KDE')
    util.plot.plt.show()
