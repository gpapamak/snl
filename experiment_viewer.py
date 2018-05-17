from itertools import izip

import os
import numpy as np
import matplotlib.pyplot as plt

import inference.mcmc as mcmc

import util.plot
import util.io
import util.math

import experiment_descriptor as ed
import misc


class ExperimentViewer:
    """
    Shows the results of a previously run experiment.
    """

    def __init__(self, exp_desc):
        """
        :param exp_desc: an experiment descriptor object
        """

        assert isinstance(exp_desc, ed.ExperimentDescriptor)

        self.exp_desc = exp_desc
        self.exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        self.sim = misc.get_simulator(exp_desc.sim)

    def print_log(self, trial=0):
        """
        Prints the log of the experiment.
        """

        print '\n' + '-' * 80
        print 'PRINTING LOG:\n'
        print self.exp_desc.pprint()

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(self.exp_desc)

        assert util.io.load_txt(os.path.join(exp_dir, 'info.txt')) == self.exp_desc.pprint()

        print util.io.load_txt(os.path.join(exp_dir, 'out.log'))

    def view_results(self, trial=0, block=False):
        """
        Shows the results of the experiment.
        :param block: whether to block execution after showing results
        """

        print '\n' + '-' * 80
        print 'VIEWING RESULTS:\n'
        print self.exp_desc.pprint()

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if not os.path.exists(exp_dir):
            raise misc.NonExistentExperiment(self.exp_desc)

        assert util.io.load_txt(os.path.join(exp_dir, 'info.txt')) == self.exp_desc.pprint()

        if isinstance(self.exp_desc.inf, ed.ABC_Descriptor):
            self._view_abc(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.SynthLik_Descriptor):
            self._view_synth_lik(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.NDE_Descriptor):
            self._view_nde(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.PostProp_Descriptor):
            self._view_post_prop(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.SNPE_MDN_Descriptor):
            self._view_snpe_mdn(exp_dir)

        elif isinstance(self.exp_desc.inf, ed.SNL_Descriptor):
            self._view_snl(exp_dir)

        else:
            raise TypeError('unknown inference descriptor')

        plt.show(block=block)

    def _view_abc(self, exp_dir):
        """
        View the results for ABC,
        """

        inf_desc = self.exp_desc.inf
        true_ps, _ = util.io.load(os.path.join(exp_dir, 'gt'))
        results = util.io.load(os.path.join(exp_dir, 'results'))

        if isinstance(inf_desc, ed.Rej_ABC_Descriptor):

            samples, dist, n_sims = results

            # posterior histogram
            fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('Rej ABC, eps = {0:.2}'.format(inf_desc.eps))

            # distances histogram
            fig, ax = plt.subplots(1, 1)
            dist = dist[dist < get_dist_disp_lim(self.exp_desc.sim)]
            ax.hist(dist, bins='auto', normed=True)
            ax.set_xlim([0, ax.get_xlim()[1]])
            ax.set_ylim([0, ax.get_ylim()[1]])
            ax.vlines(inf_desc.eps, 0, ax.get_ylim()[1], color='r')
            ax.set_xlabel('distances')
            ax.set_title('Rej ABC, eps = {0:.2}'.format(inf_desc.eps))

        elif isinstance(inf_desc, ed.MCMC_ABC_Descriptor):

            samples, _ = results

            # posterior histogram
            fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('MCMC ABC, eps = {0:.2}, step = {1:.2}'.format(inf_desc.eps, inf_desc.step))

            # trace plot
            fig = util.plot.plot_traces(samples)
            fig.suptitle('MCMC ABC, eps = {0:.2}, step = {1:.2}'.format(inf_desc.eps, inf_desc.step))

        elif isinstance(inf_desc, ed.SMC_ABC_Descriptor):

            all_samples, all_log_weights, all_eps, all_log_ess, all_n_sims = results

            # posterior histograms
            skip = max(1, len(all_eps) / 5)
            for samples, log_weights, eps in izip(all_samples[:-1:skip], all_log_weights[:-1:skip], all_eps[:-1:skip]):
                fig = util.plot.plot_hist_marginals(samples, np.exp(log_weights), lims=self.sim.get_disp_lims(), gt=true_ps)
                fig.suptitle('SMC ABC, eps = {0:.2}'.format(eps))
            fig = util.plot.plot_hist_marginals(all_samples[-1], np.exp(all_log_weights[-1]), lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('SMC ABC, eps = {0:.2}'.format(all_eps[-1]))

            # effective sample size vs iteration
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.exp(all_log_ess) * 100, ':o')
            ax.plot(ax.get_xlim(), 0.5 * np.ones(2) * 100, 'r--')
            ax.set_xlabel('iteration')
            ax.set_ylabel('effective sample size [%]')
            ax.set_title('SMC ABC')

            # sims vs eps
            fig, ax = plt.subplots(1, 1)
            ax.plot(all_eps, all_n_sims, ':o')
            ax.set_xlabel('eps')
            ax.set_ylabel('# sims')
            ax.set_title('SMC ABC')

        else:
            raise TypeError('unknown ABC descriptor')

    def _view_synth_lik(self, exp_dir):
        """
        View the results for synthetic likelihood,
        """

        inf_desc = self.exp_desc.inf
        true_ps, _ = util.io.load(os.path.join(exp_dir, 'gt'))
        samples, n_sims = util.io.load(os.path.join(exp_dir, 'results'))

        print 'Number of simulations = {0}'.format(n_sims)

        # posterior histogram
        fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('Synth likelihood, mcmc = {0}, n_sims = {1}'.format(inf_desc.mcmc.get_id(' '), inf_desc.n_sims))

        # trace plot
        fig = util.plot.plot_traces(samples)
        fig.suptitle('Synth likelihood, mcmc = {0}, n_sims = {1}'.format(inf_desc.mcmc.get_id(' '), inf_desc.n_sims))

    def _view_nde(self, exp_dir):
        """
        Shows the posterior learnt by the model in NDE.
        """

        model_desc = self.exp_desc.inf.model
        target = self.exp_desc.inf.target
        true_ps, obs_xs = self.sim.get_ground_truth()
        model = util.io.load(os.path.join(exp_dir, 'model'))

        if target == 'posterior':

            if isinstance(model_desc, ed.MDN_Descriptor):

                levels = [0.68, 0.95, 0.99]
                posterior = model.get_mog(obs_xs)
                fig = util.plot.plot_pdf_marginals(posterior, lims=self.sim.get_disp_lims(), gt=true_ps, levels=levels)

            elif isinstance(model_desc, ed.MAF_Descriptor):

                n_samples = 1000
                samples = model.gen(obs_xs, n_samples)
                fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)

            else:
                raise TypeError('unknown model type')

        elif target == 'likelihood':

            n_samples = 1000
            prior = self.sim.Prior()
            log_posterior = lambda t: model.eval([t, obs_xs]) + prior.eval(t)
            sampler = mcmc.SliceSampler(true_ps, log_posterior)
            sampler.gen(200, logger=None)  # burn in
            samples = sampler.gen(n_samples, logger=None)

            fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)

        else:
            raise ValueError('unknown target')

        fig.suptitle('NDE, ' + target + ', ' + model_desc.get_id(' '))

    def _view_post_prop(self, exp_dir):
        """
        Shows the results of learning the posterior with proposal.
        """

        model_id = self.exp_desc.inf.model.get_id(' ')

        true_ps, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))
        all_proposals, posterior, _, all_xs = util.io.load(os.path.join(exp_dir, 'results'))

        # show distances
        all_dist = [np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1)) for xs in all_xs]
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(all_dist)
        ax.set_xlabel('round')
        ax.set_title('PostProp, {0}, distances'.format(model_id))

        # show proposals
        for i, proposal in enumerate(all_proposals[1:]):
            fig = util.plot.plot_pdf_marginals(proposal, lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('PostProp, {0}, proposal round {1}'.format(model_id, i+1))

        # show posterior
        fig = util.plot.plot_pdf_marginals(posterior, lims=self.sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('PostProp, {0}, posterior'.format(model_id))

    def _view_snpe_mdn(self, exp_dir):
        """
        Shows the results of Sequential Neural Posterior Estimation with an SVI MDN.
        """

        model_id = self.exp_desc.inf.model.get_id(' ')

        true_ps, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))
        all_posteriors, _, all_xs, _ = util.io.load(os.path.join(exp_dir, 'results'))

        # show distances
        all_dist = [np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1)) for xs in all_xs]
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(all_dist)
        ax.set_xlabel('round')
        ax.set_title('SNPE MDN, {0}, distances'.format(model_id))

        # show proposals
        for i, posterior in enumerate(all_posteriors[1:]):
            fig = util.plot.plot_pdf_marginals(posterior, lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('SNPE MDN, {0}, posterior round {1}'.format(model_id, i+1))

    def _view_snl(self, exp_dir):
        """
        Shows the results of learning the likelihood with MCMC.
        """

        model_id = self.exp_desc.inf.model.get_id(' ')
        train_on = self.exp_desc.inf.train_on

        true_ps, obs_xs = util.io.load(os.path.join(exp_dir, 'gt'))
        model = util.io.load(os.path.join(exp_dir, 'model'))
        all_ps, all_xs, _ = util.io.load(os.path.join(exp_dir, 'results'))

        # show distances
        all_dist = [np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1)) for xs in all_xs]
        fig, ax = plt.subplots(1, 1)
        ax.boxplot(all_dist)
        ax.set_xlabel('round')
        ax.set_title('SNL on {0}, {1}, distances'.format(train_on, model_id))

        # show proposed parameters
        for i, ps in enumerate(all_ps):
            fig = util.plot.plot_hist_marginals(ps, lims=self.sim.get_disp_lims(), gt=true_ps)
            fig.suptitle('SNL on {0}, {1}, proposed params round {2}'.format(train_on, model_id, i+1))

        # show posterior
        n_samples = 1000
        prior = self.sim.Prior()
        log_posterior = lambda t: model.eval([t, obs_xs]) + prior.eval(t)
        sampler = mcmc.SliceSampler(true_ps, log_posterior)
        sampler.gen(200, logger=None)  # burn in
        samples = sampler.gen(n_samples, logger=None)
        fig = util.plot.plot_hist_marginals(samples, lims=self.sim.get_disp_lims(), gt=true_ps)
        fig.suptitle('SNL on {0}, {1}, posterior samples (slice sampling)'.format(train_on, model_id))


def get_dist_disp_lim(sim_desc):
    """
    Given a simulator descriptor, returns the upper display limit for the distances histogram.
    """

    dist_disp_lim = {
        'gauss': float('inf'),
        'mg1': 1.0,
        'lotka_volterra': float('inf'),
        'hodgkin_huxley': float('inf')
    }

    return dist_disp_lim[sim_desc]
