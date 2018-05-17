import sys
import numpy as np
from scipy.misc import logsumexp
from copy import deepcopy

import ml.trainers as trainers
import ml.models.mdns as mdns
import ml.step_strategies as ss
import ml.loss_functions as lf

import pdfs
import simulators
import inference.mcmc as mcmc


def learn_conditional_density(model, xs, ys, ws=None, regularizer=None, val_frac=0.05, step=ss.Adam(a=1.e-4), minibatch=100, patience=20, monitor_every=1, logger=sys.stdout, rng=np.random):
    """
    Train model to learn the conditional density p(y|x).
    """

    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)

    n_data = xs.shape[0]
    assert ys.shape[0] == n_data, 'wrong sizes'

    # shuffle data, so that training and validation sets come from the same distribution
    idx = rng.permutation(n_data)
    xs = xs[idx]
    ys = ys[idx]

    # split data into training and validation sets
    n_trn = int(n_data - val_frac * n_data)
    xs_trn, xs_val = xs[:n_trn], xs[n_trn:]
    ys_trn, ys_val = ys[:n_trn], ys[n_trn:]

    if ws is None:

        # train model without weights
        trainer = trainers.SGD(
            model=model,
            trn_data=[xs_trn, ys_trn],
            trn_loss=model.trn_loss if regularizer is None else model.trn_loss + regularizer,
            trn_target=model.y,
            val_data=[xs_val, ys_val],
            val_loss=model.trn_loss,
            val_target=model.y,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger
        )

    else:

        # prepare weights
        ws = np.asarray(ws, np.float32)
        assert ws.size == n_data, 'wrong sizes'
        ws = ws[idx]
        ws_trn, ws_val = ws[:n_trn], ws[n_trn:]

        # train model with weights
        trainer = trainers.WeightedSGD(
            model=model,
            trn_data=[xs_trn, ys_trn],
            trn_losses=-model.L,
            trn_weights=ws_trn,
            trn_reg=regularizer,
            trn_target=model.y,
            val_data=[xs_val, ys_val],
            val_losses=-model.L,
            val_weights=ws_val,
            val_target=model.y,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            patience=patience,
            monitor_every=monitor_every,
            logger=logger
        )

    return model


class PosteriorLearnerWithProposal:
    """
    Implementation of the algorithm by:
    Papamakarios & Murray, "Fast epsilon-free inference of simulation models with Bayesian conditional density estimation", NIPS 2016.
    """

    def __init__(self, prior, sim_model, n_hiddens, act_fun):

        self.prior = prior
        self.sim_model = sim_model

        self.proposal = prior
        self.all_proposals = [prior]
        self.posterior = None

        self.mdn_prop = None
        self.mdn_post = None

        self.n_hiddens = n_hiddens
        self.act_fun = act_fun

        self.all_ps = []
        self.all_xs = []

    def learn_proposal(self, obs_xs, n_samples, n_rounds, maxepochs=1000, lreg=0.01, minibatch=50, step=ss.Adam(), store_sims=False, logger=sys.stdout, rng=np.random):
        """
        Iteratively trains an bayesian MDN to learn a gaussian proposal.
        """

        # TODO: deal with tuning maxepochs

        # create mdn, if haven't already
        if self.mdn_prop is None:
            self.mdn_prop = mdns.MDN_SVI(
                n_inputs=len(obs_xs),
                n_outputs=self.prior.n_dims,
                n_hiddens=self.n_hiddens,
                act_fun=self.act_fun,
                n_components=1,
                rng=rng
            )

        for i in xrange(n_rounds):

            logger.write('Learning proposal, round {0}\n'.format(i + 1))

            # simulate new batch of data
            ps, xs = self._sim_data(n_samples, store_sims, logger, rng)

            # train mdn
            self._train_mdn(ps, xs, self.mdn_prop, maxepochs, lreg, min(minibatch, n_samples), step, logger)

            try:
                # calculate the proposal
                self.proposal = self._calc_posterior(self.mdn_prop, obs_xs).project_to_gaussian()
                self.all_proposals.append(self.proposal)

            except pdfs.gaussian.ImproperCovarianceError:
                logger.write('WARNING: learning proposal failed in iteration {0} due to negative variance.\n'.format(i+1))
                break

        return self.proposal

    def learn_posterior(self, obs_xs, n_samples, n_comps, maxepochs=5000, lreg=0.01, minibatch=100, step=ss.Adam(), store_sims=False, logger=sys.stdout, rng=np.random):
        """
        Trains a Bayesian MDN to learn the posterior using the proposal.
        """

        # TODO: deal with tuning maxepochs

        # create an svi mdn
        if self.mdn_prop is None:
            self.mdn_post = mdns.MDN_SVI(
                n_inputs=len(obs_xs),
                n_outputs=self.prior.n_dims,
                n_hiddens=self.n_hiddens,
                act_fun=self.act_fun,
                n_components=n_comps,
                rng=rng
            )

        else:
            self.mdn_post = mdns.replicate_gaussian_mdn(self.mdn_prop, n_comps, rng=rng)

        logger.write('Learning posterior\n')

        # simulate data
        ps, xs = self._sim_data(n_samples, store_sims, logger, rng)

        # train mdn
        self._train_mdn(ps, xs, self.mdn_post, maxepochs, lreg, min(minibatch, n_samples), step, logger)

        try:
            # calculate the approximate posterior
            self.posterior = self._calc_posterior(self.mdn_post, obs_xs)

        except pdfs.gaussian.ImproperCovarianceError:
            logger.write('WARNING: learning posterior failed due to negative variance.\n')
            self.posterior = self.proposal

        return self.posterior

    def _sim_data(self, n_samples, store_sims, logger, rng):
        """
        Simulates a given number of samples from the simulator.
        """

        if self.proposal is self.prior:
            trunc_proposal = self.proposal
        else:
            trunc_proposal = pdfs.TruncatedPdf(self.proposal, lambda p: self.prior.eval(p) > -float('inf'))

        logger.write('simulating data... ')
        ps, xs = simulators.sim_data(trunc_proposal.gen, self.sim_model, n_samples, rng=rng)
        logger.write('done\n')

        if store_sims:
            self.all_ps.append(ps)
            self.all_xs.append(xs)

        return ps, xs

    @staticmethod
    def _train_mdn(ps, xs, mdn, maxepochs, lreg, minibatch, step, logger):
        """
        Train SVI MDN on parameter/data samples.
        """

        ps = np.asarray(ps, np.float32)
        xs = np.asarray(xs, np.float32)

        n_samples = ps.shape[0]
        assert xs.shape[0] == n_samples, 'wrong sizes'

        regularizer = lf.SviRegularizer(mdn.mps, mdn.sps, lreg) / n_samples

        logger.write('training model...\n')

        trainer = trainers.SGD(
            model=mdn,
            trn_data=[xs, ps],
            trn_loss=mdn.trn_loss + regularizer,
            trn_target=mdn.y,
            step=step
        )
        trainer.train(
            minibatch=minibatch,
            maxepochs=maxepochs,
            monitor_every=1,
            logger=logger
        )

        logger.write('training model done\n')

        return mdn

    def _calc_posterior(self, mdn, obs_xs):
        """
        Given a trained MDN, calculates and returns the posterior at the observed data.
        """

        mog = mdn.get_mog(obs_xs, n_samples=None)
        mog.prune_negligible_components(1.0e-6)

        if self.proposal is self.prior:
            posterior = mog

        elif isinstance(self.prior, pdfs.Gaussian):
            posterior = (mog * self.prior) / self.proposal

        elif isinstance(self.prior, pdfs.Uniform):
            posterior = mog / self.proposal

        else:
            raise TypeError('algorithm only works with uniform or gaussian priors')

        return posterior


class SequentialNeuralPosteriorEstimation_MDN:
    """
    An implementation of SNPE as described by:
    Lueckmann et al., "Flexible statistical inference for mechanistic models of neural dynamics", NIPS 2017
    """

    def __init__(self, prior, sim_model, n_hiddens, act_fun, n_comps, lreg=0.01):

        self.prior = prior
        self.sim_model = sim_model

        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_comps = n_comps
        self.lreg = lreg

        self.mdn = None
        self.regularizer = None

        self.posterior = prior
        self.all_posteriors = [prior]

        self.all_ps = []
        self.all_xs = []
        self.all_ws = []

    def learn_posterior(self, obs_xs, n_samples, n_rounds, maxepochs=1000, minibatch=100, step=ss.Adam(),
                        normalize_weights=True, store_sims=False, logger=sys.stdout, rng=np.random):
        """
        Sequentially trains an SVI MDN to learn the posterior. Previous posteriors guide simulations.
        Simulated data are importance weighted when retraining the model.
        """

        # create an svi mdn
        if self.mdn is None:

            self.mdn = mdns.MDN_SVI(
                n_inputs=len(obs_xs),
                n_outputs=self.prior.n_dims,
                n_hiddens=self.n_hiddens,
                act_fun=self.act_fun,
                n_components=self.n_comps,
                rng=rng
            )

            self.regularizer = lf.SviRegularizer(self.mdn.mps, self.mdn.sps, self.lreg)

        for i in xrange(n_rounds):

            logger.write('Learning posterior, round {0}\n'.format(i + 1))

            # simulate data
            logger.write('simulating data... ')
            ps, xs = simulators.sim_data(self.posterior.gen, self.sim_model, n_samples, rng=rng)
            logger.write('done\n')

            # importance weights
            if normalize_weights:
                log_ws = self.prior.eval(ps) - self.posterior.eval(ps)
                ws = n_samples * np.exp(log_ws - logsumexp(log_ws))
            else:
                ws = np.exp(self.prior.eval(ps) - self.posterior.eval(ps))

            if store_sims:
                self.all_ps.append(ps)
                self.all_xs.append(xs)
                self.all_ws.append(ws)

            # train model
            logger.write('training model...\n')

            trainer = trainers.WeightedSGD(
                model=self.mdn,
                trn_data=[xs, ps],
                trn_losses=-self.mdn.L,
                trn_weights=ws,
                trn_reg=self.regularizer / n_samples,
                trn_target=self.mdn.y,
                step=step,
                max_norm=0.1
            )
            trainer.train(
                minibatch=minibatch,
                maxepochs=maxepochs,
                monitor_every=1,
                logger=logger
            )

            logger.write('training model done\n')

            # update regularizer
            m0s = [mp.get_value() for mp in self.mdn.mps]
            s0s = [sp.get_value() for sp in self.mdn.sps]
            self.regularizer = lf.SviRegularizer_DiagCov(self.mdn.mps, self.mdn.sps, m0s, s0s)

            self.posterior = self.mdn.get_mog(obs_xs)
            self.all_posteriors.append(self.posterior)

        return self.posterior


class SequentialNeuralLikelihood:
    """
    Trains a likelihood model using posterior MCMC sampling to guide simulations.
    """

    def __init__(self, prior, sim_model):

        self.prior = prior
        self.sim_model = sim_model

        self.all_ps = None
        self.all_xs = None
        self.all_models = None

    def learn_likelihood(self, obs_xs, model, n_samples, n_rounds, train_on_all=True, thin=10, save_models=False, logger=sys.stdout, rng=np.random):
        """
        :param obs_xs: the observed data
        :param model: the model to train
        :param n_samples: number of simulated samples per round
        :param n_rounds: number of rounds
        :param train_on_all: whether to train on all simulated samples or just on the latest batch
        :param thin: number of samples to thin the chain
        :param logger: logs messages
        :param rng: random number generator
        :return: the trained model
        """

        self.all_ps = []
        self.all_xs = []
        self.all_models = []

        log_posterior = lambda t: model.eval([t, obs_xs]) + self.prior.eval(t)
        sampler = mcmc.SliceSampler(self.prior.gen(), log_posterior, thin=thin)

        for i in xrange(n_rounds):

            logger.write('Learning likelihood, round {0}\n'.format(i + 1))

            if i == 0:
                # sample from prior in first round
                proposal = self.prior
            else:
                # MCMC sample posterior in every other round
                logger.write('burning-in MCMC chain...\n')
                sampler.gen(max(200 / thin, 1), logger=logger, rng=rng)  # burn in
                logger.write('burning-in done...\n')
                proposal = sampler

            # run new batch of simulations
            logger.write('simulating data... ')
            ps, xs = simulators.sim_data(proposal.gen, self.sim_model, n_samples, rng=rng)
            logger.write('done\n')
            self.all_ps.append(ps)
            self.all_xs.append(xs)

            if train_on_all:
                ps = np.concatenate(self.all_ps)
                xs = np.concatenate(self.all_xs)

            N = ps.shape[0]
            monitor_every = min(10 ** 5 / float(N), 1.0)

            # retrain likelihood model
            logger.write('training model...\n')
            learn_conditional_density(model, ps, xs, monitor_every=monitor_every, logger=logger, rng=rng)
            logger.write('training done\n')

            if save_models:
                self.all_models.append(deepcopy(model))

        return model
