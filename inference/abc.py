"""
Implementations of ABC algorithms for likelihood-free simulation-based inference.
"""

import os
import sys
import numpy as np
import scipy.misc

import util.math


def calc_dist(data_1, data_2):
    """
    Calculates the distance between two data vectors. Here the euclidean distance is used.
    """

    if data_1 is None or data_2 is None:
        return float('inf')

    diff = data_1 - data_2
    dist = np.sqrt(np.dot(diff, diff))

    return dist


class Rejection:
    """
    Implements rejection abc.
    """

    def __init__(self, prior, sim_model):

        self.prior = prior
        self.sim_model = sim_model

    def run(self, obs_data, eps, n_samples, logger=sys.stdout, info=False, rng=np.random):

        ps = []
        dist = []
        n_sims = 0
        n_accepted = 0

        logger = open(os.devnull, 'w') if logger is None else logger

        while n_accepted < n_samples:

            prop_ps = self.prior.gen(rng=rng)
            prop_data = self.sim_model(prop_ps, rng=rng)
            prop_dist = calc_dist(prop_data, obs_data)
            n_sims += 1

            if prop_dist < eps:
                ps.append(prop_ps)
                n_accepted += 1

            if info:
                dist.append(prop_dist)

            logger.write('sim = {0}, accepted = {1}, dist = {2:.3}, acc rate = {3:.2%}\n'.format(n_sims, n_accepted, prop_dist, float(n_accepted) / n_sims))

        if info:
            return np.array(ps), np.array(dist), n_sims
        else:
            return np.array(ps)


class MCMC:
    """
    Implements markov chain monte carlo for abc.
    """

    def __init__(self, prior, sim_model, init_ps):

        self.prior = prior
        self.sim_model = sim_model
        self.cur_ps = np.asarray(init_ps, float)
        self.cur_log_prior = self.prior.eval(self.cur_ps, log=True)

    def run(self, obs_data, eps, step, n_samples, logger=sys.stdout, info=False, rng=np.random):
        """
        Runs mcmc abc. Uses a spherical gaussian proposal.
        """

        ps = []
        n_accepted = 0
        cur_dist = None
        n_dim = self.cur_ps.size

        logger = open(os.devnull, 'w') if logger is None else logger

        for i in xrange(n_samples):

            prop_ps = self.cur_ps + step * rng.randn(n_dim)
            prop_data = self.sim_model(prop_ps, rng=rng)
            prop_dist = calc_dist(prop_data, obs_data)

            # acceptance / rejection step
            if prop_dist < eps:

                prop_log_prior = self.prior.eval(prop_ps, log=True)

                if rng.rand() < np.exp(prop_log_prior - self.cur_log_prior):

                    self.cur_ps = prop_ps
                    self.cur_log_prior = prop_log_prior
                    cur_dist = prop_dist
                    n_accepted += 1

            ps.append(self.cur_ps.copy())

            logger.write('iter = {0}, dist = {1:.3}, acc rate = {2:.2%}\n'.format(i, cur_dist, float(n_accepted) / (i+1)))

        ps = np.array(ps)
        acc_rate = float(n_accepted) / n_samples

        if info:
            return ps, acc_rate
        else:
            return ps


class SMC:
    """
    Implements sequential monte carlo for abc.
    """

    def __init__(self, prior, sim_model):

        self.prior = prior
        self.sim_model = sim_model

    def run(self, obs_data, eps_init, eps_last, eps_decay, n_particles, ess_min=0.5, logger=sys.stdout, info=False, rng=np.random):
        """
        Runs full smc abc.
        """

        all_ps = []
        all_log_weights = []
        all_eps = []
        all_log_ess = []
        all_n_sims = []

        logger = open(os.devnull, 'w') if logger is None else logger

        # save some log values for reuse
        log_ess_min = np.log(ess_min)
        log_n_particles = np.log(n_particles)

        # sample initial population
        iter = 0
        eps = eps_init
        ps, n_sims = self.sample_initial_population(obs_data, n_particles, eps, logger, rng)
        log_weights = np.full(n_particles, -log_n_particles)

        if info:
            all_ps.append(ps)
            all_log_weights.append(log_weights)
            all_eps.append(eps)
            all_log_ess.append(0.0)
            all_n_sims.append(n_sims)

        logger.write('iter = {0}, eps = {1:.2}, ess = {2:.2%}, sims = {3}\n'.format(iter, eps, 1.0, n_sims))

        while eps > eps_last:

            # sample next population
            iter += 1
            eps *= eps_decay
            ps, log_weights, n_new_sims = self.sample_next_population(ps, log_weights, obs_data, eps, logger, rng)
            n_sims += n_new_sims

            # calculate effective sample size
            log_ess = -scipy.misc.logsumexp(2.0 * log_weights) - log_n_particles

            # if population is degenerate, resample particles
            if log_ess < log_ess_min:
                ps = self.resample_population(ps, log_weights, rng)
                log_weights = np.full(n_particles, -log_n_particles)

            if info:
                all_ps.append(ps)
                all_log_weights.append(log_weights)
                all_eps.append(eps)
                all_log_ess.append(log_ess)
                all_n_sims.append(n_sims)

            logger.write('iter = {0}, eps = {1:.2}, ess = {2:.2%}, sims = {3}\n'.format(iter, eps, np.exp(log_ess), n_sims))

        if info:
            return all_ps, all_log_weights, all_eps, all_log_ess, all_n_sims
        else:
            return ps, log_weights

    def sample_initial_population(self, obs_data, n_particles, eps, logger, rng):
        """
        Sample an initial population of n_particles, with tolerance eps.
        """

        ps = []
        n_sims = 0

        for i in xrange(n_particles):

            dist = float('inf')
            prop_ps = None

            while dist > eps:
                prop_ps = self.prior.gen(rng=rng)
                data = self.sim_model(prop_ps, rng=rng)
                dist = calc_dist(data, obs_data)
                n_sims += 1

            ps.append(prop_ps)

            logger.write('particle {0}\n'.format(i + 1))

        return np.array(ps), n_sims

    def sample_next_population(self, ps, log_weights, obs_data, eps, logger, rng):
        """
        Samples a new population of particles by perturbing an existing one. Uses a gaussian perturbation kernel.
        """

        n_particles, n_dim = ps.shape
        n_sims = 0
        weights = np.exp(log_weights)

        # calculate population covariance
        mean = np.mean(ps, axis=0)
        cov = 2.0 * (np.dot(ps.T, ps) / n_particles - np.outer(mean, mean))
        std = np.linalg.cholesky(cov)

        new_ps = np.empty_like(ps)
        new_log_weights = np.empty_like(log_weights)

        for i in xrange(n_particles):

            dist = float('inf')

            while dist > eps:
                idx = util.math.discrete_sample(weights, rng=rng)
                new_ps[i] = ps[idx] + np.dot(std, rng.randn(n_dim))
                data = self.sim_model(new_ps[i], rng=rng)
                dist = calc_dist(data, obs_data)
                n_sims += 1

            # calculate unnormalized weights
            log_kernel = -0.5 * np.sum(scipy.linalg.solve_triangular(std, (new_ps[i] - ps).T, lower=True) ** 2, axis=0)
            new_log_weights[i] = self.prior.eval(new_ps[i], log=True) - scipy.misc.logsumexp(log_weights + log_kernel)

            logger.write('particle {0}\n'.format(i + 1))

        # normalize weights
        new_log_weights -= scipy.misc.logsumexp(new_log_weights)

        return new_ps, new_log_weights, n_sims

    @staticmethod
    def resample_population(ps, log_weights, rng):
        """
        Resample an existing population of particles.
        """

        n_particles = ps.shape[0]
        idx = util.math.discrete_sample(np.exp(log_weights), n_particles, rng=rng)
        ps = ps[idx]

        return ps
