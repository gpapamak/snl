import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import util.io
import util.plot

from plot_results_loglik import get_sim, ed, get_samples_snl
from plot_results_mmd import get_true_samples


def make_plots(samples, sim, sim_name):
    """
    Makes all plots.
    """

    n_dim = samples.shape[1]
    n_bins = int(np.sqrt(samples.shape[0]))
    lims = np.asarray(sim.get_disp_lims())
    lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims
    true_ps, _ = sim.get_ground_truth()

    if sim_name == 'hh':
        ii = sim.Model().ps_order_in_plot
        true_ps = true_ps[ii]
        samples = samples[:, ii]
        lims = lims[ii]

    for i in xrange(n_dim):
        for j in xrange(i, n_dim):

            if i == j:

                fig, ax = plt.subplots(1, 1)
                ax.hist(samples[:, i], bins=n_bins, normed=True, rasterized=True)
                ax.set_xlim(lims[i])
                ax.set_ylim([0.0, ax.get_ylim()[1]])
                ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.vlines(true_ps[i], 0, ax.get_ylim()[1], color='r')

            else:

                fig, ax = plt.subplots(1, 1)
                ax.scatter(samples[:, j], samples[:, i], c='k', s=3, marker='o', cmap='binary', edgecolors='none', rasterized=True)
                ax.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
                ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.set_xlim(lims[j])
                ax.set_ylim(lims[i])
                ax.scatter(true_ps[j], true_ps[i], c='r', s=12, marker='o', edgecolors='none')


def plot_results(sim_name):
    """
    Plots posterior for a given simulator.
    """

    sim = get_sim(sim_name)

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_seq.txt'.format(sim_name))):

        if isinstance(exp_desc.inf, ed.SNL_Descriptor):

            samples = get_samples_snl(exp_desc, sim)[-1]
            make_plots(samples, sim, sim_name)

    if sim_name == 'gauss':

        samples = get_true_samples()
        make_plots(samples, sim, sim_name)

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting the posteriors for SNL.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    args = parser.parse_args()

    plot_results(args.sim)


if __name__ == '__main__':
    main()
