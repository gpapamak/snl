import numpy as np
import matplotlib.pyplot as plt
import experiment_descriptor as ed


class NonExistentExperiment(Exception):
    """
    Exception to be thrown when the requested experiment doesn't exist.
    """

    def __init__(self, exp_desc):
        assert isinstance(exp_desc, ed.ExperimentDescriptor)
        self.exp_desc = exp_desc

    def __str__(self):
        return self.exp_desc.pprint()


class AlreadyExistingExperiment(Exception):
    """
    Exception to be thrown when the requested experiment already exists.
    """

    def __init__(self, exp_desc):
        assert isinstance(exp_desc, ed.ExperimentDescriptor)
        self.exp_desc = exp_desc

    def __str__(self):
        return self.exp_desc.pprint()


def get_simulator(sim_desc):
    """
    Given the description of a simulator, returns the simulator module.
    """

    if sim_desc == 'mg1':
        import simulators.mg1 as sim

    elif sim_desc == 'lotka_volterra':
        import simulators.lotka_volterra as sim

    elif sim_desc == 'gauss':
        import simulators.gaussian as sim

    elif sim_desc == 'hodgkin_huxley':
        import simulators.hodgkin_huxley as sim

    else:
        raise ValueError('unknown simulator')

    return sim


def get_root():

    return 'data'


def find_epsilon(sims_loader, obs_xs, acc_rate, show_hist=True):
    """
    Finds an epsilon for rejection ABC that yeilds a particular acceptance rate.
    :param sims_loader: a simulations loader object
    :param obs_xs: the observed data
    :param acc_rate: acceptance rate
    :param show_hist: if True, show a histogram of the distances
    :return: epsilon value
    """

    n_sims = 10**6
    xs = None

    while True:

        try:
            _, xs = sims_loader.load(n_sims)
            break

        except RuntimeError:
            n_sims /= 2

    dist = np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1))
    eps = np.percentile(dist, acc_rate * 100)

    if show_hist:
        fig, ax = plt.subplots(1, 1)
        ax.hist(dist, bins='auto', normed=True)
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.vlines(eps, 0, ax.get_ylim()[1], color='r')
        ax.set_xlabel('distances')
        plt.show()

    return eps
