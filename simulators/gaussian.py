import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import simulators
import pdfs
import util.plot
import util.misc


class Prior(pdfs.BoxUniform):
    """
    A uniform prior over m1, m2, (+/-)sqrt(s1), (+/-)sqrt(s2), arctanh(r).
    """

    def __init__(self):

        n_dims = 5
        lower = [-3.0] * n_dims
        upper = [+3.0] * n_dims
        pdfs.BoxUniform.__init__(self, lower, upper)


class Model(simulators.SimulatorModel):
    """
    Simulator model.
    """

    def __init__(self):

        simulators.SimulatorModel.__init__(self)
        self.n_data = 4

    def sim(self, ps, rng=np.random):
        """
        Simulate data at parameters ps.
        """

        ps = np.asarray(ps, float)

        if ps.ndim == 1:
            return self.sim(ps[np.newaxis, :], rng=rng)[0]

        n_sims = ps.shape[0]

        m0, m1, s0, s1, r = self._unpack_params(ps)

        us = rng.randn(n_sims, self.n_data, 2)
        xs = np.empty_like(us)

        xs[:, :, 0] = s0 * us[:, :, 0] + m0
        xs[:, :, 1] = s1 * (r * us[:, :, 0] + np.sqrt(1.0 - r ** 2) * us[:, :, 1]) + m1

        self.n_sims += n_sims

        return xs.reshape([n_sims, 2*self.n_data])

    def eval(self, px, log=True):
        """
        Evaluate probability of data given parameters.
        """

        ps, xs, one_datapoint = util.misc.prepare_cond_input(px, float)

        m0, m1, s0, s1, r = self._unpack_params(ps)
        logdet = np.log(s0) + np.log(s1) + 0.5 * np.log(1.0 - r ** 2)

        xs = xs.reshape([xs.shape[0], self.n_data, 2])
        us = np.empty_like(xs)

        us[:, :, 0] = (xs[:, :, 0] - m0) / s0
        us[:, :, 1] = (xs[:, :, 1] - m1 - s1 * r * us[:, :, 0]) / (s1 * np.sqrt(1.0 - r ** 2))
        us = us.reshape([us.shape[0], 2 * self.n_data])

        L = np.sum(scipy.stats.norm.logpdf(us), axis=1) - self.n_data * logdet[:, 0]
        L = L[0] if one_datapoint else L

        return L if log else np.exp(L)

    @staticmethod
    def _unpack_params(ps):
        """
        Unpack parameters ps to m0, m1, s0, s1, r.
        """

        assert ps.shape[1] == 5, 'wrong size'

        m0 = ps[:, [0]]
        m1 = ps[:, [1]]
        s0 = ps[:, [2]] ** 2
        s1 = ps[:, [3]] ** 2
        r = np.tanh(ps[:, [4]])

        return m0, m1, s0, s1, r


class Stats:
    """
    Identity summary stats.
    """

    def __init__(self):
        pass

    @staticmethod
    def calc(ps):
        return ps


class SimsLoader(simulators.RealTimeSimsLoader):
    """
    Loads existing simulation data. Uses singleton pattern to load data only once.
    """

    def __init__(self):

        simulators.RealTimeSimsLoader.__init__(self, Prior(), Model(), Stats())


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_ps = [-0.7, -2.9, -1.0, -0.9, 0.6]

    rng = np.random.RandomState(41)
    obs_xs = Stats().calc(Model().sim(true_ps, rng=rng))

    return true_ps, obs_xs


def get_ground_truth_sims(n_samples):
    """
    Returns a given number of simulated statistics from true parameters.
    """

    rng = np.random.RandomState(42)

    model = Model()
    stats = Stats()
    true_ps, _ = get_ground_truth()

    xs = stats.calc(model.sim(np.tile(true_ps, [n_samples, 1]), rng=rng))

    return xs


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-5.0, 5.0]


def test_simulator():

    n_sims = 1000

    xs = get_ground_truth_sims(n_sims)
    xs = xs.reshape([n_sims, -1, 2])

    true_ps, _ = get_ground_truth()
    m = true_ps[:2]

    for i in xrange(xs.shape[1]):

        x = xs[:, i, :]
        util.plot.plot_hist_marginals(x, gt=m)

    plt.show()


def do_mcmc_inference(rng=np.random):

    import inference.mcmc as mcmc

    true_ps, obs_xs = get_ground_truth()

    prior = Prior()
    model = Model()
    log_posterior = lambda t: model.eval([t, obs_xs]) + prior.eval(t)

    sampler = mcmc.SliceSampler(true_ps, log_posterior)
    ps = sampler.gen(10000, show_info=True, rng=rng)

    util.plot.plot_hist_marginals(ps, lims=get_disp_lims(), gt=true_ps)
    plt.show()
