import os
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import misc

import inference.diagnostics.two_sample as two_sample
import experiment_descriptor as ed
import pdfs
import util.io
import util.math
import util.plot

root = misc.get_root()

n_samples = 5000

g_true_ps = None
g_true_samples = None
g_scale = None

sim = None


def get_truth():
    """
    Returns the true params, samples from the true params, and a scale for mmd.
    Saves results in global variables so as to do work only once.
    """

    global g_true_ps, g_true_samples, g_scale

    if g_true_ps is None or g_true_samples is None or g_scale is None:

        g_true_ps, _ = sim.get_ground_truth()
        g_true_samples = sim.get_ground_truth_sims(n_samples)
        g_scale = util.math.median_distance(g_true_samples)

    return g_true_ps, g_true_samples, g_scale


def get_disp_lims():
    """
    Returns display limits for likelihood samples calculated from true samples.
    """

    _, true_samples, _ = get_truth()

    lims = np.array([np.min(true_samples, axis=0), np.max(true_samples, axis=0)])
    diff = lims[1] - lims[0]
    lims[0] -= 0.1 * diff
    lims[1] += 0.1 * diff
    lims = lims.T

    return lims


def view_true_samples(use_lims=True):
    """
    Plots samples from the true likelihood.
    """

    _, true_samples, _ = get_truth()
    _, obs_xs = sim.get_ground_truth()

    lims = get_disp_lims() if use_lims else None

    fig = util.plot.plot_hist_marginals(true_samples, lims=lims, gt=obs_xs)
    fig.suptitle('true samples')

    plt.plot()


def view_samples_nde(sim_name, which=None, use_lims=True):
    """
    Plots likelihood samples for all NDE models.
    """

    true_ps, obs_xs = sim.get_ground_truth()

    lims = get_disp_lims() if use_lims else None

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_nde.txt'.format(sim_name))):

        if which is not None and exp_desc.inf.n_samples != which:
            continue

        exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')
        net = util.io.load(os.path.join(exp_dir, 'model'))
        samples = net.gen(true_ps, n_samples)

        fig = util.plot.plot_hist_marginals(samples, lims=lims, gt=obs_xs)
        fig.suptitle('NDE, sims = {0}'.format(exp_desc.inf.n_samples))

    plt.plot()


def view_samples_snl(sim_name, which=None, use_lims=True):
    """
    Plots likelihood samples for all SNL models.
    """

    true_ps, obs_xs = sim.get_ground_truth()

    lims = get_disp_lims() if use_lims else None

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_prop.txt'.format(sim_name))):

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):

            exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')
            _, _, all_nets = util.io.load(os.path.join(exp_dir, 'results'))

            for i, net in enumerate(all_nets):

                if which is not None and i != which - 1:
                    continue

                net.reset_theano_functions()
                samples = net.gen(true_ps, n_samples)

                fig = util.plot.plot_hist_marginals(samples, lims=lims, gt=obs_xs)
                fig.suptitle('SNL, round = {0}'.format(i + 1))

    plt.plot()


def calc_mmd(model):
    """
    Calculates MMD between true samples and a given likelihood model.
    """

    _, true_samples, scale = get_truth()
    samples = model.gen(true_samples.shape[0], rng=np.random.RandomState(42))

    return two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)


def calc_mmd_cond(net):
    """
    Calculates MMD between true samples and a given conditional likelihood model.
    """

    true_ps, true_samples, scale = get_truth()
    samples = net.gen(true_ps, true_samples.shape[0], rng=np.random.RandomState(42))

    return two_sample.sq_maximum_mean_discrepancy(samples, true_samples, scale=scale)


def get_err_gaussian(sim_name):
    """
    Calculates the error for a gaussian fit.
    """

    res_file = os.path.join(root, 'results', translate_sim_name(sim_name), 'other', 'gaussian_lik_mmd')

    if os.path.exists(res_file + '.pkl'):
        err = util.io.load(res_file)

    else:
        _, true_samples, _ = get_truth()
        gauss = pdfs.fit_gaussian(true_samples)
        err = calc_mmd(gauss)
        util.io.save(err, res_file)

    return err


def get_err_nde(exp_desc):
    """
    Calculates the error for a given NDE experiment.
    """

    assert isinstance(exp_desc.inf, ed.NDE_Descriptor)
    res_file = os.path.join(root, 'results', exp_desc.get_dir(), 'lik_mmd')

    if os.path.exists(res_file + '.pkl'):
        err = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        net = util.io.load(os.path.join(exp_dir, 'model'))
        err = calc_mmd_cond(net)
        util.io.save(err, res_file)

    return err


def get_err_snl(exp_desc):
    """
    Calculates the error for a given SNL experiment.
    """

    assert isinstance(exp_desc.inf, ed.SNL_Descriptor)
    res_file = os.path.join(root, 'results', exp_desc.get_dir(), 'lik_mmd')

    if os.path.exists(res_file + '.pkl'):
        all_errs = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(exp_desc)

        _, _, all_nets = util.io.load(os.path.join(exp_dir, 'results'))
        all_errs = []

        for net in all_nets:
            net.reset_theano_functions()
            all_errs.append(calc_mmd_cond(net))

        util.io.save(all_errs, res_file)

    return all_errs


def translate_sim_name(sim_name):
    """
    Translates the simulator name from its abbreviation to the full name.
    """

    if sim_name == 'lv':
        return 'lotka_volterra'

    elif sim_name == 'hh':
        return 'hodgkin_huxley'

    else:
        return sim_name


def plot_results(sim_name):
    """
    Plots all results for a given simulator and kind of error.
    """

    global sim
    sim = misc.get_simulator(translate_sim_name(sim_name))

    # gaussian
    err_gauss = get_err_gaussian(sim_name)
    err_gauss = max(err_gauss, 0.0)

    # NDE
    all_err_nde = []
    all_n_sims_nde = []
    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_nde.txt'.format(sim_name))):
        all_err_nde.append(get_err_nde(exp_desc))
        all_n_sims_nde.append(exp_desc.inf.n_samples)

    all_err_snl = None
    all_n_sims_snl = None

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_prop.txt'.format(sim_name))):

        # SNL
        if isinstance(exp_desc.inf, ed.SNL_Descriptor):
            all_err_snl = get_err_snl(exp_desc)
            all_n_sims_snl = [(i + 1) * exp_desc.inf.n_samples for i in xrange(exp_desc.inf.n_rounds)]

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    all_n_sims = np.concatenate([all_n_sims_nde, all_n_sims_snl])
    min_n_sims = np.min(all_n_sims)
    max_n_sims = np.max(all_n_sims)

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(all_n_sims_nde, np.sqrt(all_err_nde), 's:', color='b', label='NL')
    ax.semilogx(all_n_sims_snl, np.sqrt(all_err_snl), 'o:', color='r', label='SNL')
    ax.axhline(np.sqrt(err_gauss), linestyle='--', color='k', label='Gaussian')
    ax.set_xlabel('Number of simulations (log scale)')
    ax.set_ylabel('MMD')
    ax.set_xlim([min_n_sims * 10 ** (-0.2), max_n_sims * 10 ** 0.2])
    ax.set_ylim([-0.1 if sim_name == 'gauss' else 0.0, ax.get_ylim()[1]])
    ax.legend(fontsize=14, loc='upper right' if sim_name == 'gauss' else 'lower right')

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting the results for the likelihood goodness of fit experiment.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    args = parser.parse_args()

    plot_results(args.sim)


if __name__ == '__main__':
    main()
