import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import neuron
from neuron import h

import simulators

import pdfs

import util.io
import util.math
import util.plot


# create the neuron
h.load_file('stdrun.hoc')
h.load_file('sPY_template')
h('objref IN')
h('IN = new sPY()')
h.celsius = 36

# create electrode
h('objref El')
h('IN.soma El = new IClamp(0.5)')
h('El.del = 0')
h('El.dur = 100')

# set simulation time and initial voltage
h.tstop = 100.0
h.v_init = -70.0

# record voltage
h('objref v_vec')
h('objref t_vec')
h('t_vec = new Vector()')
h('t_vec.indgen(0, tstop, dt)')
h('v_vec = new Vector()')
h('IN.soma v_vec.record(&v(0.5), t_vec)')


class Prior(pdfs.BoxUniform):
    """
    A uniform prior for the Hodgkin-Huxley model.
    """

    def __init__(self):

        ref = np.log([0.0001, 0.2, 0.05, 7e-05, 70.0, 50.0, 100.0, 60.0, 0.5, 40.0, 1000.0, 1.0])
        lower = ref - np.log(2.0)
        upper = lower + np.log(3.0)
        pdfs.BoxUniform.__init__(self, lower, upper)


class Model(simulators.SimulatorModel):
    """
    Hodgkin-Huxley simulator.
    """

    ps_names = ['g_leak', 'gbar_Na', 'gbar_K', 'gbar_M', 'E_leak', 'E_Na', 'E_K', 'V_T',
                'k_betan1', 'k_betan2', 'tau_max', 'sigma']

    ps_order_in_plot = [1, 2, 0, 5, 6, 4, 3, 10, 8, 9, 7, 11]

    def __init__(self):

        simulators.SimulatorModel.__init__(self)

    def sim(self, ps, rng=np.random):
        """
        Run the simulation with the given params.
        """

        ps = np.asarray(ps, float)
        ps = np.exp(ps)

        if ps.ndim == 1:
            return self._sim_one(ps, rng=rng)
        else:
            return np.array([self._sim_one(p, rng=rng) for p in ps])

    def _sim_one(self, ps, rng):
        """
        Run the simulation for one setting of parameters.
        """

        # set parameters
        h.IN.soma[0](0.5).g_pas = ps[0]        # g_leak
        h.IN.soma[0](0.5).gnabar_hh2 = ps[1]   # gbar_Na
        h.IN.soma[0](0.5).gkbar_hh2 = ps[2]    # gbar_K
        h.IN.soma[0](0.5).gkbar_im = ps[3]     # gbar_M
        h.IN.soma[0](0.5).e_pas = -ps[4]       # E_leak
        h.IN.soma[0](0.5).ena = ps[5]          # E_Na
        h.IN.soma[0](0.5).ek = -ps[6]          # E_K
        h.IN.soma[0](0.5).vtraub_hh2 = -ps[7]  # V_T
        h.IN.soma[0](0.5).kbetan1_hh2 = ps[8]  # k_betan1
        h.IN.soma[0](0.5).kbetan2_hh2 = ps[9]  # k_betan2
        h.taumax_im = ps[10]                   # tau_max
        sigma = ps[11]                         # sigma

        # set up current injection of noise
        Iinj = rng.normal(0.5, sigma, np.array(h.t_vec).size)
        Iinj_vec = h.Vector(Iinj)
        Iinj_vec.play(h.El._ref_amp, h.t_vec)

        # initialize and run
        neuron.init()
        h.finitialize(h.v_init)
        neuron.run(h.tstop)

        self.n_sims += 1

        return np.array(h.v_vec)


class Stats:
    """
    Summary statistics for the Hodgkin-Huxley model.
    """

    def __init__(self):

        self.whiten_params = util.io.load(os.path.join(get_root(), 'stats_whitening_params'))

    def calc(self, data):

        data = np.asarray(data, float)

        if data.ndim == 1:
            return self.calc(data[np.newaxis, :])[0]

        n_data, n_dims = data.shape
        eps = 1.0e-7

        xs = np.empty([n_data, 18])
        i = 0

        # mean
        mean = np.mean(data, axis=1)
        xs[:, i] = mean
        i += 1

        # std
        std = np.std(data, axis=1)
        xs[:, i] = np.log(std + eps)
        i += 1

        # normalize
        data = (data - mean[:, np.newaxis]) / std[:, np.newaxis]

        # even moments
        for m in [4, 6, 8]:
            xs[:, i] = np.log(scipy.stats.moment(data, m, axis=1) + eps)
            i += 1

        # odd moments
        for m in [3, 5, 7]:
            xs[:, i] = scipy.stats.moment(data, m, axis=1)
            i += 1

        # auto correlations
        for lag in 100 * (np.arange(10) + 1):
            xs[:, i] = np.sum(data[:, :-lag] * data[:, lag:], axis=1) / n_dims
            i += 1

        # whiten
        xs = util.math.whiten(xs, self.whiten_params)

        assert i == xs.shape[1]

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

    return 'data/simulators/hodgkin_huxley'


def get_disp_lims():
    """
    Returns the parameter display limits.
    """

    prior = Prior()

    return np.stack([prior.lower, prior.upper]).T


def get_ground_truth():
    """
    Returns ground truth parameters and corresponding observed statistics.
    """

    return util.io.load(os.path.join(get_root(), 'obs_data'))


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


def demo1(use_gt=True, rng=np.random):
    """
    Runs and plots one simulation.
    :param use_gt: whether to use the ground truth or a random parameter setting
    """

    if use_gt:
        ps, _ = get_ground_truth()
    else:
        ps = Prior().gen(rng=rng)

    vs = Model().sim(ps, rng=rng)
    ts = h.dt * np.arange(vs.size)

    fig, ax = plt.subplots(1, 1)
    ax.plot(ts, vs)
    ax.set_xlabel('time')
    ax.set_ylabel('membrane potential')

    plt.show()


def demo2(rng=np.random):
    """
    Verify that changing some parameters influences the output.
    """

    model = Model()
    ps, _ = get_ground_truth()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('time')
    ax.set_ylabel('membrane potential')

    ps[11] = np.log(10.0)  # sigma
    vs = model.sim(ps, rng=rng)
    ts = h.dt * np.arange(vs.size)
    ax.plot(ts, vs, color='red', label='sigma = {0:.2}, gbar_Na = {1:.2}'.format(np.exp(ps[11]), np.exp(ps[1])))

    ps[11] = np.log(0.5)  # sigma
    vs = model.sim(ps, rng=rng)
    ax.plot(ts, vs, color='blue', label='sigma = {0:.2}, gbar_Na = {1:.2}'.format(np.exp(ps[11]), np.exp(ps[1])))

    ps[1] += np.log(4.0)  # gbar_Na
    vs = model.sim(ps, rng=rng)
    ax.plot(ts, vs, color='orange', label='sigma = {0:.2}, gbar_Na = {1:.2}'.format(np.exp(ps[11]), np.exp(ps[1])))

    ax.legend()
    plt.show()


def demo3():
    """
    Verify that changing every parameter influences the output.
    """

    model = Model()

    plt.figure()

    ps0, _ = get_ground_truth()
    ps0[11] = np.log(0.01)  # sigma
    ps0[7] = np.log(60.0)   # -V_T
    vs0 = model.sim(ps0, rng=np.random.RandomState(42))
    ts = h.dt * np.arange(vs0.size)

    dps = np.exp(ps0)
    dps[11] = 10.0   # sigma
    dps[2] = 1.0     # gbar_K
    dps[8] = 100.0   # k_betan1
    dps[9] = 100.0   # k_betan2

    for i in xrange(12):

        ps = np.array(ps0)
        ps[i] = np.log(dps[i] + np.exp(ps[i]))
        vs = model.sim(ps, rng=np.random.RandomState(42))

        plt.subplot(4, 3, i + 1)
        plt.title(model.ps_names[i])
        plt.plot(ts, vs0, color='blue')
        plt.plot(ts, vs, color='red')

    plt.show()


def demo4(rng=np.random):
    """
    Plots random simulations with parameters drawn from the prior.
    """

    prior = Prior()
    model = Model()

    vs = model.sim(prior.gen(30), rng=rng)
    ts = h.dt * np.arange(vs.shape[1])

    fig, axs = plt.subplots(6, 5)

    for v, ax in zip(vs, axs.flatten()):
        ax.plot(ts, v)

    plt.show()


def demo5(rng=np.random):
    """
    Plots random simulations and their summary stats.
    """

    prior = Prior()
    model = Model()
    stats = Stats()

    vs = model.sim(prior.gen(12), rng=rng)
    ts = h.dt * np.arange(vs.shape[1])
    xs = stats.calc(vs)

    fig, axs = plt.subplots(6, 2)

    for v, x, ax in zip(vs, xs, axs):
        ax[0].plot(ts, v)
        ax[1].bar(np.arange(x.size) + 1, x)

    plt.show()


def demo6(n_samples=1000):
    """
    Shows histograms of parameter and statistics.
    """

    true_ps, obs_xs = get_ground_truth()
    ps, xs = SimsLoader().load(n_samples)

    # plot prior parameter histograms
    fig = util.plot.plot_hist_marginals(ps, lims=get_disp_lims(), gt=true_ps)
    fig.suptitle('p(thetas)')

    # plot stats histograms
    fig = util.plot.plot_hist_marginals(util.math.de_whiten(xs, Stats().whiten_params), gt=obs_xs)
    fig.suptitle('p(stats), not whitened')

    # plot stats histograms, whitened
    fig = util.plot.plot_hist_marginals(xs, gt=obs_xs)
    fig.suptitle('p(stats), whitened')

    plt.show()
