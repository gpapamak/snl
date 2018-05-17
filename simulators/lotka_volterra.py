import os
import numpy as np
import matplotlib.pyplot as plt

import pdfs
import simulators
import simulators.markov_jump_processes as mjp
import util.io
import util.plot


class Prior(pdfs.BoxUniform):
    """
    A uniform prior for the Lotka-Volterra model.
    """

    def __init__(self):

        lower = np.full(4, -5.0)
        upper = np.full(4, +2.0)
        pdfs.BoxUniform.__init__(self, lower, upper)


class PriorNearTruth(pdfs.TruncatedPdf):
    """
    A gaussian prior centred at the true parameters, truncated by the uniform prior.
    """

    def __init__(self, std=0.5):

        true_ps, _ = get_ground_truth()
        prior = Prior()

        gaussian = pdfs.Gaussian(m=true_ps, S=std**2 * np.eye(prior.n_dims))
        indicator = lambda p: prior.eval(p) > -float('inf')

        pdfs.TruncatedPdf.__init__(self, gaussian, indicator)


class Model(simulators.SimulatorModel):
    """
    The Lotka-Volterra model.
    """

    def __init__(self):

        simulators.SimulatorModel.__init__(self)

        self.init = [50, 100]
        self.dt = 0.2
        self.duration = 30
        self.max_n_steps = 10000
        self.lv = mjp.LotkaVolterra(self.init, None)

    def sim(self, ps, remove_failed=False, rng=np.random):

        ps = np.asarray(ps, float)
        ps = np.exp(ps)

        if ps.ndim == 1:

            self.n_sims += 1

            try:
                self.lv.reset(self.init, ps)
                states = self.lv.sim_time(self.dt, self.duration, max_n_steps=self.max_n_steps, rng=rng)
                return states.flatten()

            except mjp.SimTooLongException:
                return None

        elif ps.ndim == 2:

            data = []

            for p in ps:

                try:
                    self.lv.reset(self.init, p)
                    states = self.lv.sim_time(self.dt, self.duration, max_n_steps=self.max_n_steps, rng=rng)
                    data.append(states.flatten())

                except mjp.SimTooLongException:
                    data.append(None)

            if remove_failed:
                data = filter(lambda u: u is not None, data)

            self.n_sims += ps.shape[0]

            return np.array(data)

        else:
            raise ValueError('wrong size')


class Stats:
    """
    Summary statistics for the Lotka-Volterra model.
    """

    def __init__(self):

        self.means, self.stds = util.io.load(os.path.join(get_root(), 'pilot_run_results'))

    def calc(self, data):

        if data is None:
            return None

        data = np.asarray(data)
        has_nones = np.any(map(lambda u: u is None, data))

        if data.ndim == 1 and not has_nones:
            return self._calc_one_datapoint(data)

        else:
            return np.array([None if x is None else self._calc_one_datapoint(x) for x in data])

    def _calc_one_datapoint(self, data):

        xy = np.reshape(data, [-1, 2])
        x, y = xy[:, 0], xy[:, 1]
        N = xy.shape[0]

        # means
        mx = np.mean(x)
        my = np.mean(y)

        # variances
        s2x = np.var(x, ddof=1)
        s2y = np.var(y, ddof=1)

        # standardize
        x = (x - mx) / np.sqrt(s2x)
        y = (y - my) / np.sqrt(s2y)

        # auto correlation coefficient
        acx = []
        acy = []
        for lag in [1, 2]:
            acx.append(np.dot(x[:-lag], x[lag:]) / (N-1))
            acy.append(np.dot(y[:-lag], y[lag:]) / (N-1))

        # cross correlation coefficient
        ccxy = np.dot(x, y) / (N-1)

        # normalize stats
        xs = np.array([mx, my, np.log(s2x + 1.0), np.log(s2y + 1.0)] + acx + acy + [ccxy])
        xs -= self.means
        xs /= self.stds

        return xs


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

    return 'data/simulators/lotka_volterra'


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    return [-5., 2.]


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    true_ps = np.log([0.01, 0.5, 1.0, 0.01])
    obs_xs = util.io.load(os.path.join(get_root(), 'obs_stats'))

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


def plot_one_sim(use_gt=True):
    """
    Runs and plots a single simulation of the Lotka-Volterra model.
    """

    if use_gt:
        ps, _ = get_ground_truth()
    else:
        ps = Prior().gen()

    model = Model()
    data = model.sim(ps)
    if data is None:
        print 'Simulation failed'
        return
    data = np.reshape(data, [-1, 2])
    times = np.linspace(0., model.duration, int(model.duration / model.dt) + 1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(times, data[:, 0], label='predators')
    ax.plot(times, data[:, 1], label='prey')
    ax.set_xlabel('time')
    ax.set_ylabel('population counts')
    ax.legend()

    fig, ax = plt.subplots(1, 1)
    ax.plot(data[:, 0], data[:, 1])
    ax.set_xlabel('predators')
    ax.set_ylabel('prey')

    plt.show()


def plot_many_sims(n_sims=1000):
    """
    Runs many simulations and plots histograms.
    """

    true_ps, obs_xs = get_ground_truth()
    ps, xs = SimsLoader().load(n_sims)

    # parameter histogram
    fig = util.plot.plot_hist_marginals(ps, lims=get_disp_lims(), gt=true_ps)
    fig.suptitle('parameters')

    # data histogram
    fig = util.plot.plot_hist_marginals(xs, gt=obs_xs)
    fig.suptitle('summary statistics')

    plt.show()


def plot_sims_near_truth(std=0.5):
    """
    Plots random simulations from a gaussian proposal centred at true params with given stdev.
    """

    true_ps, _ = get_ground_truth()
    proposal = PriorNearTruth(std)
    model = Model()

    _, xs = simulators.sim_data(proposal.gen, model.sim, 30)
    fig, axs = plt.subplots(6, 5)
    times = np.linspace(0., model.duration, int(model.duration / model.dt) + 1)

    for data, ax in zip(xs, axs.flatten()):

        data = np.reshape(data, [-1, 2])

        ax.plot(times, data[:, 0])
        ax.plot(times, data[:, 1])
        ax.set_xlabel('time')
        ax.set_ylabel('population counts')

    util.plot.plot_pdf_marginals(proposal.pdf, lims=get_disp_lims(), gt=true_ps).suptitle('proposal')

    plt.show()
