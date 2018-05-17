import os
import numpy as np
import matplotlib.pyplot as plt

import simulators
import pdfs
import util.plot
import util.io
import util.math


class Prior(pdfs.Uniform):
    """
    A uniform non-separable prior for the M/G/1 model.
    """

    def __init__(self):

        pdfs.Uniform.__init__(self, 3)
        self.uniform = pdfs.BoxUniform([0., 0., 0.], [10., 10., 1./3.])

        self.A = np.array([[1., -1., 0.], [0., 1., 0.], [0., 0., 1.]])
        self.A_inv = np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])

    def _to_u(self, x):

        return np.dot(x, self.A)

    def _to_p(self, x):

        return np.dot(x, self.A_inv)

    def eval(self, ps, log=True):

        ps = np.asarray(ps, float)
        assert (ps.ndim == 1 and ps.shape[0] == 3) or (ps.ndim == 2 and ps.shape[1] == 3), 'wrong size'
        return self.uniform.eval(self._to_u(ps), log=log)

    def gen(self, n_samples=None, rng=np.random):

        return self._to_p(self.uniform.gen(n_samples, rng))

    def show_histograms(self, n_samples=1000):
        """
        Generates samples and plots their histograms.
        """

        ps = self.gen(n_samples)
        util.plot.plot_hist_marginals(ps, lims=get_disp_lims())
        plt.show()


class Model(simulators.SimulatorModel):
    """
    The M/G/1 queue model.
    """

    def __init__(self):

        simulators.SimulatorModel.__init__(self)
        self.n_sim_steps = 50

    def sim(self, ps, info=False, rng=np.random):

        ps = np.asarray(ps, float)

        if ps.ndim == 1:

            res = self.sim(ps[np.newaxis, :], info, rng)
            return tuple(map(lambda x: x[0], res)) if info else res[0]

        elif ps.ndim == 2:

            assert ps.shape[1] == 3, 'parameter must be 3-dimensional'
            p1, p2, p3 = ps[:, 0:1], ps[:, 1:2], ps[:, 2:3]
            N = ps.shape[0]

            # service times (uniformly distributed)
            sts = (p2 - p1) * rng.rand(N, self.n_sim_steps) + p1

            # inter-arrival times (exponentially distributed)
            iats = -np.log(1.0 - rng.rand(N, self.n_sim_steps)) / p3

            # arrival times
            ats = np.cumsum(iats, axis=1)

            # inter-departure times
            idts = np.empty([N, self.n_sim_steps], dtype=float)
            idts[:, 0] = sts[:, 0] + ats[:, 0]

            # departure times
            dts = np.empty([N, self.n_sim_steps], dtype=float)
            dts[:, 0] = idts[:, 0]

            for i in xrange(1, self.n_sim_steps):
                idts[:, i] = sts[:, i] + np.maximum(0.0, ats[:, i] - dts[:, i-1])
                dts[:, i] = dts[:, i-1] + idts[:, i]

            self.n_sims += N

            return (sts, iats, ats, idts, dts) if info else idts

        else:
            raise TypeError('parameters must be either a 1-dim or a 2-dim array')


class Stats:
    """
    Summary statistics for the M/G/1 model: percentiles of the inter-departure times
    """

    def __init__(self):

        n_percentiles = 5
        self.perc = np.linspace(0.0, 100.0, n_percentiles)

        self.whiten_params = util.io.load(os.path.join(get_root(), 'pilot_run_results'))

    def calc(self, data, whiten=True):

        data = np.asarray(data, float)

        if data.ndim == 1:
            return self.calc(data[np.newaxis, :], whiten)[0]

        elif data.ndim == 2:

            stats = np.percentile(data, self.perc, axis=1).T

            if whiten:
                stats = util.math.whiten(stats, self.whiten_params)

            return stats

        else:
            raise TypeError('data must be either a 1-dim or a 2-dim array')


class SimsLoader(simulators.RealTimeSimsLoader):
    """
    Loads existing simulation data. Uses singleton pattern to load data only once.
    """

    def __init__(self):

        # to save computation, save simulations to files and use simulators.StoredSimsLoader to load them
        simulators.RealTimeSimsLoader.__init__(self, Prior(), Model(), Stats(), 1000)


def get_root():
    """
    Returns the root folder.
    """

    return 'data/simulators/mg1'


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [[0., 10.], [0., 20.], [0., 1./3.]]


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_ps, obs_xs = util.io.load(os.path.join(get_root(), 'observed_data'))
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


def test_mg1():
    """
    Runs and plots a single simulation of the model.
    """

    ps = [1.0, 5.0, 0.2]
    # ps = Prior().gen()

    model = Model()
    sts, iats, ats, idts, dts = model.sim(ps, info=True)

    stats = Stats().calc(idts)
    print stats

    times, sizes = calc_size_of_queue(ats, dts)
    fig, ax = plt.subplots(1, 1)
    ax.plot(times, sizes, drawstyle='steps')
    ax.set_xlabel('time')
    ax.set_ylabel('queue size')
    ax.set_title('ps = {0:.2f}, {1:.2f}, {2:.2f}'.format(*ps))

    n_bins = int(np.sqrt(model.n_sim_steps))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.hist(sts, bins=n_bins, normed=True)
    ax1.set_xlabel('service times')
    ax2.hist(iats, bins=n_bins, normed=True)
    ax2.set_xlabel('iter-arrival times')
    ax3.hist(idts, bins=n_bins, normed=True)
    ax3.set_xlabel('inter-departure times')
    fig.suptitle('ps = {0:.2f}, {1:.2f}, {2:.2f}'.format(*ps))

    plt.show()


def calc_size_of_queue(ats, dts):
    """
    Given arrival and departure times, calculates size of queue at any time.
    """

    N = len(ats)
    assert len(dts) == N

    ats_inf = np.append(ats, float('inf'))
    dts_inf = np.append(dts, float('inf'))

    times = [0.0]
    sizes = [0]

    i = 0
    j = 0

    while i < N or j < N:

        # new arrival
        if ats_inf[i] < dts_inf[j]:
            times.append(ats[i])
            sizes.append(sizes[-1] + 1)
            i += 1

        # new departure
        elif ats_inf[i] > dts_inf[j]:
            times.append(dts[j])
            sizes.append(sizes[-1] - 1)
            j += 1

        # simultaneous arrival and departure
        else:
            i += 1
            j += 1

    assert np.all(np.array(sizes) >= 0)

    return times, sizes


def show_histograms(n_samples=1000):
    """
    Simulates from joint and shows histograms of simulations.
    """

    true_ps, obs_xs = get_ground_truth()

    prior = Prior()
    model = Model()
    stats = Stats()

    ps = prior.gen(n_samples)
    data = stats.calc(model.sim(ps))
    cond_data = stats.calc(model.sim(np.tile(true_ps, [n_samples, 1])))

    # plot prior parameter histograms
    fig = util.plot.plot_hist_marginals(ps, lims=get_disp_lims(), gt=true_ps)
    fig.suptitle('p(thetas)')

    # plot stats histograms
    fig = util.plot.plot_hist_marginals(data, gt=obs_xs)
    fig.suptitle('p(stats)')

    # plot stats histograms, conditioned on true params
    fig = util.plot.plot_hist_marginals(cond_data, gt=obs_xs)
    fig.suptitle('p(stats|true thetas)')

    plt.show()
