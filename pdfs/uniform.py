import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import util.plot


class Uniform:
    """
    Parent class for uniform distributions.
    """

    def __init__(self, n_dims):

        self.n_dims = n_dims

    def grad_log_p(self, x):
        """
        :param x: rows are datapoints
        :return: d/dx log p(x)
        """

        x = np.asarray(x)
        assert (x.ndim == 1 and x.size == self.n_dims) or (x.ndim == 2 and x.shape[1] == self.n_dims), 'wrong size'

        return np.zeros_like(x)


class BoxUniform(Uniform):
    """
    Implements a uniform pdf, constrained in a box.
    """

    def __init__(self, lower, upper):
        """
        :param lower: array with lower limits
        :param upper: array with upper limits
        """

        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        assert lower.ndim == 1 and upper.ndim == 1 and lower.size == upper.size, 'wrong sizes'
        assert np.all(lower < upper), 'invalid upper and lower limits'

        Uniform.__init__(self, lower.size)

        self.lower = lower
        self.upper = upper
        self.volume = np.prod(upper - lower)

    def eval(self, x, ii=None, log=True):
        """
        :param x: evaluate at rows
        :param ii: a list of indices to evaluate marginal, if None then evaluates joint
        :param log: whether to return the log prob
        :return: the prob at x rows
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        if ii is None:

            in_box = np.logical_and(self.lower <= x, x <= self.upper)
            in_box = np.logical_and.reduce(in_box, axis=1)

            if log:
                prob = -float('inf') * np.ones(in_box.size, dtype=float)
                prob[in_box] = -np.log(self.volume)

            else:
                prob = np.zeros(in_box.size, dtype=float)
                prob[in_box] = 1.0 / self.volume

            return prob

        else:
            assert len(ii) > 0, 'list of indices can''t be empty'
            marginal = BoxUniform(self.lower[ii], self.upper[ii])
            return marginal.eval(x, None, log)

    def gen(self, n_samples=None, rng=np.random):
        """
        :param n_samples: int, number of samples to generate 
        :return: numpy array, rows are samples. Only 1 sample (vector) if None
        """

        one_sample = n_samples is None
        u = rng.rand(1 if one_sample else n_samples, self.n_dims)
        x = (self.upper - self.lower) * u + self.lower
        
        return x[0] if one_sample else x


def test_uniform_3d():

    lims = np.array([[0, -1, 2], [1, 1, 5]])
    pdf = BoxUniform(lims[0], lims[1])

    x = 0.5 * (pdf.lower + pdf.upper)
    print 'volume =', pdf.volume
    print 'height =', pdf.eval(x, log=False)
    print 'log height =', pdf.eval(x, log=True)

    xs = pdf.gen(10000)
    assert np.all(pdf.eval(xs, log=False) > 0.0), 'BUG: generated samples at zero probability region'

    disp_lims = (lims + [[-1, -1, -1], [1, 1, 1]]).T
    util.plot.plot_pdf_marginals(pdf, disp_lims, levels=[1.0 - 1.0e-12])
    util.plot.plot_hist_marginals(xs, lims=disp_lims)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], 'k.', ms=2)

    plt.show()
