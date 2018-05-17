import numpy as np
import pdfs
import util.misc
import util.io


def sim_data(gen_params, sim_model, n_samples=None, rng=np.random):
    """
    Simulates a and returns a given number of samples from the simulator.
    Takes care of failed simulations, and guarantees the exact number of requested samples will be returned.
    If number of samples is None, it returns one sample.
    """

    if n_samples is None:
        ps, xs = sim_data(gen_params, sim_model, n_samples=1, rng=rng)
        return ps[0], xs[0]

    assert n_samples > 0

    ps = None
    xs = None

    while True:

        # simulate parameters and data
        ps = gen_params(n_samples, rng=rng)
        xs = sim_model(ps, rng=rng)

        # filter out simulations that failed
        idx = [x is not None for x in xs]

        if not np.any(idx):
            continue

        if not np.all(idx):
            ps = np.stack(ps[idx])
            xs = np.stack(xs[idx])

        break  # we'll break only when we have at least one successful simulation

    n_rem = n_samples - ps.shape[0]
    assert n_rem < n_samples

    if n_rem > 0:
        # request remaining simulations
        ps_rem, xs_rem = sim_data(gen_params, sim_model, n_rem, rng)
        ps = np.concatenate([ps, ps_rem], axis=0)
        xs = np.concatenate([xs, xs_rem], axis=0)

    assert ps.shape[0] == xs.shape[0] == n_samples

    return ps, xs


def gaussian_synthetic_likelihood(px, sim_model, log=True, n_sims=100, rng=np.random):
    """
    Calculates a gaussian synthetic likelihood of a simulator, by fitting a gaussian to simulated data.
    :param px: a parameters-data pair where to evaluate the synthetic likelihood
    :param sim_model: function that simulates data given parameters
    :param log: whether to return the log likelihood
    :param n_sims: number of datapoints to simulate
    :param rng: random number generator
    """

    ps, xs, one_datapoint = util.misc.prepare_cond_input(px, float)
    Ls = []

    for p, x in zip(ps, xs):

        xs_sim = sim_model(np.tile(p, [n_sims, 1]), rng=rng)
        xs_sim = np.array([x_sim for x_sim in xs_sim if x_sim is not None])

        if xs_sim.size == 0:
            L = -float('inf')
        else:
            L = pdfs.fit_gaussian(xs_sim, eps=1.0e-6).eval(x, log=log)

        Ls.append(L)

    Ls = np.array(Ls)

    return Ls[0] if one_datapoint else Ls


class SimulatorModel:
    """
    Base class for a simulator model.
    """

    def __init__(self):

        self.n_sims = 0

    def sim(self, ps):

        raise NotImplementedError('simulator model must be implemented as a subclass')


class RealTimeSimsLoader:
    """
    A simulations loader which runs simulations in real time.
    """

    def __init__(self, prior, model, stats, batch=10**6):

        self.ps = None
        self.xs = None

        self.prior = prior
        self.sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))

        self.batch = batch
        self.rng = np.random.RandomState(42)

    def load(self, n_samples):

        n_sims = min(self.batch, n_samples)

        if self.ps is None or self.xs is None:

            self.ps, self.xs = sim_data(self.prior.gen, self.sim_model, n_sims, rng=self.rng)

        while self.ps.shape[0] < n_samples:

            ps, xs = sim_data(self.prior.gen, self.sim_model, n_sims, rng=self.rng)
            self.ps = np.concatenate([self.ps, ps], axis=0)
            self.xs = np.concatenate([self.xs, xs], axis=0)

        return self.ps[:n_samples], self.xs[:n_samples]


class StoredSimsLoader:
    """
    A simulations loader which loads previously stored simulations.
    """

    def __init__(self, file_pattern, stats=None):

        self.ps = None
        self.xs = None

        self.n_files = 0
        self.file_pattern = file_pattern
        self.stats = stats

    def load(self, n_samples):

        try:
            if self.n_files == 0:

                self.ps, data = util.io.load(self.file_pattern.format(0))
                self.xs = data if self.stats is None else self.stats.calc(data)
                self.n_files = 1

                assert self.ps.shape[0] == self.xs.shape[0]

            while self.ps.shape[0] < n_samples:

                ps, data = util.io.load(self.file_pattern.format(self.n_files))
                xs = data if self.stats is None else self.stats.calc(data)
                self.ps = np.concatenate([self.ps, ps], axis=0)
                self.xs = np.concatenate([self.xs, xs], axis=0)
                self.n_files += 1

                assert self.ps.shape[0] == self.xs.shape[0]

        except IOError:
            raise RuntimeError('not enough data available')

        return self.ps[:n_samples], self.xs[:n_samples]
