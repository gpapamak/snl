import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import experiment_descriptor as ed
import misc

import util.io

root = misc.get_root()


def get_dist(exp_desc, average):
    """
    Get the average distance from observed data in every round.
    """

    if average == 'mean':
        fname = 'dist_obs'
        avg_f = np.mean

    elif average == 'median':
        fname = 'dist_obs_median'
        avg_f = np.median

    else:
        raise ValueError('unknown average: {0}'.format(average))

    res_file = os.path.join(root, 'results', exp_desc.get_dir(), fname)

    if os.path.exists(res_file + '.pkl'):
        avg_dist = util.io.load(res_file)

    else:
        exp_dir = os.path.join(root, 'experiments', exp_desc.get_dir(), '0')

        _, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))
        results = util.io.load(os.path.join(exp_dir, 'results'))

        if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
            _, _, _, all_xs = results

        elif isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
            _, _, all_xs, _ = results

        elif isinstance(exp_desc.inf, ed.SNL_Descriptor):
            _, all_xs, _ = results

        else:
            raise TypeError('unsupported experiment descriptor')

        avg_dist = []

        for xs in all_xs:
            dist = np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1))
            dist = filter(lambda x: not np.isnan(x), dist)
            avg_dist.append(avg_f(dist))

        util.io.save(avg_dist, res_file)

    return avg_dist


def plot_results(sim_name, average):
    """
    Plots all results for a given simulator.
    """

    all_dist_ppr = None
    all_dist_snp = None
    all_dist_snl = None

    for exp_desc in ed.parse(util.io.load_txt('exps/{0}_prop.txt'.format(sim_name))):

        # Post Prop
        if isinstance(exp_desc.inf, ed.PostProp_Descriptor):
            all_dist_ppr = get_dist(exp_desc, average)

        # SNPE
        if isinstance(exp_desc.inf, ed.SNPE_MDN_Descriptor):
            all_dist_snp = get_dist(exp_desc, average)

        # SNL
        if isinstance(exp_desc.inf, ed.SNL_Descriptor):
            all_dist_snl = get_dist(exp_desc, average)

    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', size=16)

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(len(all_dist_ppr)) + 1, all_dist_ppr, '>:', color='c', label='SNPE-A')
    ax.plot(np.arange(len(all_dist_snp)) + 1, all_dist_snp, 'p:', color='g', label='SNPE-B')
    ax.plot(np.arange(len(all_dist_snl)) + 1, all_dist_snl, 'o:', color='r', label='SNL')
    ax.set_xlabel('Round')
    ax.set_ylabel('{0} distance'.format(average[0].upper() + average[1:]))
    ax.set_ylim([0.0, ax.get_ylim()[1]])
    ax.legend(fontsize=14)

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Plotting distance vs time for the attention-focusing experiments.')
    parser.add_argument('sim', type=str, choices=['gauss', 'mg1', 'lv', 'hh'], help='simulator')
    parser.add_argument('-a', '--average', type=str, choices=['mean', 'median'], default='median', help='average type')
    args = parser.parse_args()

    plot_results(args.sim, args.average)


if __name__ == '__main__':
    main()
