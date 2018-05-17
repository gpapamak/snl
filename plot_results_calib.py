import os
import argparse
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

import inference.mcmc as mcmc

import simulators
import simulators.lotka_volterra

import experiment_descriptor as ed
import misc

import util.io


root = misc.get_root()
rng = np.random

# for mcmc
thin = 100
burnin = 5
n_samples = 9


def get_failed_sims_model_lv():

    fname = os.path.join(root, 'results', 'lotka_volterra', 'other', 'failed_sims_model', 'model')

    if not os.path.exists(fname + '.pkl'):

        import learn_failed_sims_lv
        learn_failed_sims_lv.main()

    fs_net = util.io.load(fname)

    return fs_net


def get_samples_snl(exp_desc, trial, sim, prior_kind):
    """
    Generates MCMC samples for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)

    if prior_kind == 'original':
        folder = exp_desc.get_dir()
        prior = sim.Prior()

    elif prior_kind == 'near_truth':
        folder = os.path.join(exp_desc.sim, 'other', 'prior_near_truth', exp_desc.inf.get_dir())
        prior = sim.PriorNearTruth()

    else:
        raise ValueError('unknown prior: {0}'.format(prior_kind))

    res_file = os.path.join(root, 'results', folder, str(trial), 'samples')

    if os.path.exists(res_file + '.pkl'):
        samples = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments', folder, str(trial))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        net = util.io.load(os.path.join(exp_dir, 'model'))
        true_ps, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))

        if sim is simulators.lotka_volterra:
            fs_net = get_failed_sims_model_lv()
            log_posterior = lambda t: net.eval([t, obs_xs]) + prior.eval(t) + np.log(fs_net.eval(t)[0])
        else:
            log_posterior = lambda t: net.eval([t, obs_xs]) + prior.eval(t)

        print 'sampling trial {0}'.format(trial)
        sampler = mcmc.SliceSampler(true_ps, log_posterior, thin=thin)
        sampler.gen(burnin, rng=rng)
        samples = sampler.gen(n_samples, rng=rng)

        util.io.save(samples, res_file)

    return samples


def get_order_snl(exp_desc, n_trials, n_bins, sim, prior):
    """
    Calculates the order statistic for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)

    if prior == 'original':
        folder = exp_desc.get_dir()

    elif prior == 'near_truth':
        folder = os.path.join(exp_desc.sim, 'other', 'prior_near_truth', exp_desc.inf.get_dir())

    else:
        raise ValueError('unknown prior: {0}'.format(prior))

    n_dims = sim.Prior().n_dims
    order = np.empty([n_trials, n_dims])

    for i in xrange(n_trials):

        exp_dir = os.path.join(root, 'experiments', folder, str(i + 1))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        true_ps, _ = util.io.load(os.path.join(exp_dir, 'gt'))
        samples = get_samples_snl(exp_desc, i + 1, sim, prior)
        samples = samples[:n_bins - 1]

        if samples.shape[0] < n_bins - 1:
            raise RuntimeError('not enough samples for {0} bins'.format(n_bins))

        for j in xrange(n_dims):
            order[i, j] = sum(true_ps[j] > samples[:, j])

    return order


def get_hist_quantile(prob, n_trials, n_bins):
    """
    Calculates a given quantile of the height of a bin of a uniform histogram.
    :param prob: quantile probability
    :param n_trials: number of datapoints in the histogram
    :param n_bins: number of bins in the histogram
    :return: quantile
    """

    assert 0.0 <= prob <= 1.0

    k = 0
    while scipy.stats.binom.cdf(k, n_trials, 1.0 / n_bins) < prob:
        k += 1

    return k / float(n_trials)


def get_sim(sim_name):
    """
    Returns the simulator object for a given simulator name.
    """

    if sim_name == 'lv':
        return misc.get_simulator('lotka_volterra')

    elif sim_name == 'hh':
        return misc.get_simulator('hodgkin_huxley')

    else:
        return misc.get_simulator(sim_name)


def plot_results(sim_name, prior):
    """
    Plots all results for a given simulator.
    """

    n_trials = 200
    n_bins = 10

    l_quant = get_hist_quantile(0.005, n_trials, n_bins)
    u_quant = get_hist_quantile(0.995, n_trials, n_bins)
    centre = 1.0 / n_bins

    sim = get_sim(sim_name)

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=15 if sim_name == 'lv' else 14)

    # SNL
    txt = util.io.load_txt('exps/{0}_trials.txt'.format(sim_name))

    for exp_desc in ed.parse(txt):

        order = get_order_snl(exp_desc, n_trials, n_bins, sim, prior)

        for j in xrange(order.shape[1]):

            fig, ax = plt.subplots(1, 1)
            ax.hist(order[:, j], bins=np.arange(n_bins + 1) - 0.5, normed=True, color='r')
            ax.axhspan(l_quant, u_quant, facecolor='0.5', alpha=0.5)
            ax.axhline(centre, color='k', lw=2)
            ax.set_xlim([-0.5, n_bins - 0.5])
            if sim_name == 'lv' and j != 1:
                ax.set_ylim([0.0, ax.get_ylim()[1]])
            else:
                ax.set_ylim([0.0, u_quant * 1.1])
            ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            if sim_name == 'lv':
                ax.set_title(r'$\theta_{' + str(j + 1) + r'}$, ' + ('oscillating regime' if prior == 'near_truth' else 'broad prior'))
            else:
                ax.set_title(r'$\theta_{' + str(j + 1) + r'}$')

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting the results for the diagnostic experiment.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    parser.add_argument('-p', '--prior', type=str, choices=['original', 'near_truth'], default='original', help='prior')
    args = parser.parse_args()

    plot_results(args.sim, args.prior)


if __name__ == '__main__':
    main()
