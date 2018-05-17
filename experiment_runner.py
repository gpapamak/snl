import os
import shutil
import numpy as np

import util.plot
import util.io
import util.math

import experiment_descriptor as ed
import misc

import simulators


class ExperimentRunner:
    """
    Runs experiments on likelihood-free inference of simulator models.
    """

    def __init__(self, exp_desc):
        """
        :param exp_desc: an experiment descriptor object
        """

        assert isinstance(exp_desc, ed.ExperimentDescriptor)

        self.exp_desc = exp_desc
        self.exp_dir = os.path.join(misc.get_root(), 'experiments', exp_desc.get_dir())
        self.sim = misc.get_simulator(exp_desc.sim)

    def run(self, trial=0, sample_gt=False, rng=np.random):
        """
        Runs the experiment.
        :param rng: random number generator to use
        """

        print '\n' + '-' * 80
        print 'RUNNING EXPERIMENT, TRIAL {0}:\n'.format(trial)
        print self.exp_desc.pprint()

        exp_dir = os.path.join(self.exp_dir, str(trial))

        if os.path.exists(exp_dir):
            raise misc.AlreadyExistingExperiment(self.exp_desc)

        util.io.make_folder(exp_dir)

        try:
            if isinstance(self.exp_desc.inf, ed.ABC_Descriptor):
                self._run_abc(exp_dir, sample_gt, rng)

            elif isinstance(self.exp_desc.inf, ed.SynthLik_Descriptor):
                self._run_synth_lik(exp_dir, sample_gt, rng)

            elif isinstance(self.exp_desc.inf, ed.NDE_Descriptor):
                self._train_model(exp_dir, rng)

            elif isinstance(self.exp_desc.inf, ed.PostProp_Descriptor):
                self._run_post_prop(exp_dir, sample_gt, rng)

            elif isinstance(self.exp_desc.inf, ed.SNPE_MDN_Descriptor):
                self._run_snpe_mdn(exp_dir, sample_gt, rng)

            elif isinstance(self.exp_desc.inf, ed.SNL_Descriptor):
                self._run_snl(exp_dir, sample_gt, rng)

            else:
                raise TypeError('unknown inference descriptor')

        except:
            shutil.rmtree(exp_dir)
            raise

    def _run_abc(self, exp_dir, sample_gt, rng):
        """
        Runs the ABC experiments.
        """

        import inference.abc as abc

        inf_desc = self.exp_desc.inf

        prior = self.sim.Prior()
        model = self.sim.Model()
        stats = self.sim.Stats()
        sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))
        true_ps, obs_xs = simulators.sim_data(prior.gen, sim_model, rng=rng) if sample_gt else self.sim.get_ground_truth()

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if isinstance(inf_desc, ed.Rej_ABC_Descriptor):
                abc_runner = abc.Rejection(prior, sim_model)
                results = abc_runner.run(
                    obs_xs,
                    eps=inf_desc.eps,
                    n_samples=inf_desc.n_samples,
                    logger=logger,
                    info=True,
                    rng=rng
                )

            elif isinstance(inf_desc, ed.MCMC_ABC_Descriptor):
                abc_runner = abc.MCMC(prior, sim_model, true_ps)
                results = abc_runner.run(
                    obs_xs,
                    eps=inf_desc.eps,
                    step=inf_desc.step,
                    n_samples=inf_desc.n_samples,
                    logger=logger,
                    info=True,
                    rng=rng
                )

            elif isinstance(inf_desc, ed.SMC_ABC_Descriptor):
                abc_runner = abc.SMC(prior, sim_model)
                results = abc_runner.run(
                    obs_xs,
                    eps_init=inf_desc.eps_init,
                    eps_last=inf_desc.eps_last,
                    eps_decay=inf_desc.eps_decay,
                    n_particles=inf_desc.n_samples,
                    logger=logger,
                    info=True,
                    rng=rng
                )

            else:
                raise TypeError('unknown ABC algorithm')

            util.io.save((true_ps, obs_xs), os.path.join(exp_dir, 'gt'))
            util.io.save(results, os.path.join(exp_dir, 'results'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _run_synth_lik(self, exp_dir, sample_gt, rng):
        """
        Runs gaussian synthetic likelihood.
        """

        import inference.mcmc as mcmc
        from simulators import gaussian_synthetic_likelihood

        inf_desc = self.exp_desc.inf
        mcmc_desc = inf_desc.mcmc

        prior = self.sim.Prior()
        model = self.sim.Model()
        stats = self.sim.Stats()
        sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))
        true_ps, obs_xs = simulators.sim_data(prior.gen, sim_model, rng=rng) if sample_gt else self.sim.get_ground_truth()

        log_posterior = lambda ps: gaussian_synthetic_likelihood([ps, obs_xs], sim_model, n_sims=inf_desc.n_sims, rng=rng) + prior.eval(ps)

        if isinstance(mcmc_desc, ed.GaussianMetropolisDescriptor):
            sampler = mcmc.GaussianMetropolis(true_ps, log_posterior, mcmc_desc.step)

        elif isinstance(mcmc_desc, ed.SliceSamplerDescriptor):
            sampler = mcmc.SliceSampler(true_ps, log_posterior)

        else:
            raise TypeError('unknown MCMC algorithm')

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            samples = sampler.gen(
                n_samples=mcmc_desc.n_samples,
                logger=logger,
                rng=rng
            )

            util.io.save((true_ps, obs_xs), os.path.join(exp_dir, 'gt'))
            util.io.save((samples, model.n_sims), os.path.join(exp_dir, 'results'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _train_model(self, exp_dir, rng):
        """
        Trains the model for the NDE experiments.
        """

        import inference.nde as nde

        target = self.exp_desc.inf.target
        n_samples = self.exp_desc.inf.n_samples

        ps, xs = self.sim.SimsLoader().load(n_samples)

        monitor_every = min(10 ** 5 / float(n_samples), 1.0)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            if target == 'posterior':
                model = self._create_model(xs.shape[1], ps.shape[1], rng)
                model = nde.learn_conditional_density(model, xs, ps, monitor_every=monitor_every, logger=logger, rng=rng)

            elif target == 'likelihood':
                model = self._create_model(ps.shape[1], xs.shape[1], rng)
                model = nde.learn_conditional_density(model, ps, xs, monitor_every=monitor_every, logger=logger, rng=rng)

            else:
                raise ValueError('unknown distribution')

            util.io.save(model, os.path.join(exp_dir, 'model'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _run_post_prop(self, exp_dir, sample_gt, rng):
        """
        Runs the posterior learner with proposal.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        model_desc = inf_desc.model
        assert isinstance(model_desc, ed.MDN_Descriptor)

        prior = self.sim.Prior()
        model = self.sim.Model()
        stats = self.sim.Stats()
        sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))
        true_ps, obs_xs = simulators.sim_data(prior.gen, sim_model, rng=rng) if sample_gt else self.sim.get_ground_truth()

        learner = nde.PosteriorLearnerWithProposal(prior, sim_model, model_desc.n_hiddens, model_desc.act_fun)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            learner.learn_proposal(
                obs_xs=obs_xs,
                n_samples=inf_desc.n_samples_p,
                n_rounds=inf_desc.n_rounds_p,
                maxepochs=inf_desc.maxepochs_p,
                store_sims=True,
                logger=logger,
                rng=rng
            )

            learner.learn_posterior(
                obs_xs=obs_xs,
                n_samples=inf_desc.n_samples_f,
                n_comps=model_desc.n_comps,
                maxepochs=inf_desc.maxepochs_f,
                store_sims=True,
                logger=logger,
                rng=rng
            )

            util.io.save((true_ps, obs_xs), os.path.join(exp_dir, 'gt'))
            util.io.save((learner.mdn_prop, learner.mdn_post), os.path.join(exp_dir, 'models'))
            util.io.save((learner.all_proposals, learner.posterior, learner.all_ps, learner.all_xs), os.path.join(exp_dir, 'results'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _run_snpe_mdn(self, exp_dir, sample_gt, rng):
        """
        Runs Sequential Neural Posterior Estimation with an SVI MDN.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf
        model_desc = inf_desc.model
        assert isinstance(model_desc, ed.MDN_Descriptor)

        prior = self.sim.Prior()
        model = self.sim.Model()
        stats = self.sim.Stats()
        sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))
        true_ps, obs_xs = simulators.sim_data(prior.gen, sim_model, rng=rng) if sample_gt else self.sim.get_ground_truth()

        learner = nde.SequentialNeuralPosteriorEstimation_MDN(prior, sim_model, model_desc.n_hiddens, model_desc.act_fun, model_desc.n_comps)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            learner.learn_posterior(
                obs_xs=obs_xs,
                n_samples=inf_desc.n_samples,
                n_rounds=inf_desc.n_rounds,
                maxepochs=inf_desc.maxepochs,
                store_sims=True,
                logger=logger,
                rng=rng
            )

            util.io.save((true_ps, obs_xs), os.path.join(exp_dir, 'gt'))
            util.io.save(learner.mdn, os.path.join(exp_dir, 'model'))
            util.io.save((learner.all_posteriors, learner.all_ps, learner.all_xs, learner.all_ws), os.path.join(exp_dir, 'results'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _run_snl(self, exp_dir, sample_gt, rng):
        """
        Runs the likelihood learner with MCMC.
        """

        import inference.nde as nde

        inf_desc = self.exp_desc.inf

        prior = self.sim.Prior()
        model = self.sim.Model()
        stats = self.sim.Stats()
        sim_model = lambda ps, rng: stats.calc(model.sim(ps, rng=rng))
        true_ps, obs_xs = simulators.sim_data(prior.gen, sim_model, rng=rng) if sample_gt else self.sim.get_ground_truth()

        net = self._create_model(prior.n_dims, len(obs_xs), rng)

        learner = nde.SequentialNeuralLikelihood(prior, sim_model)

        with util.io.Logger(os.path.join(exp_dir, 'out.log')) as logger:

            learner.learn_likelihood(
                obs_xs=obs_xs,
                model=net,
                n_samples=inf_desc.n_samples,
                n_rounds=inf_desc.n_rounds,
                train_on_all=(inf_desc.train_on == 'all'),
                thin=inf_desc.thin,
                save_models=True,
                logger=logger,
                rng=rng
            )

            util.io.save((true_ps, obs_xs), os.path.join(exp_dir, 'gt'))
            util.io.save(net, os.path.join(exp_dir, 'model'))
            util.io.save((learner.all_ps, learner.all_xs, learner.all_models), os.path.join(exp_dir, 'results'))
            util.io.save_txt(self.exp_desc.pprint(), os.path.join(exp_dir, 'info.txt'))

    def _create_model(self, n_inputs, n_outputs, rng):
        """
        Given input and output sizes, creates and returns the model for the NDE experiments.
        """

        model_desc = self.exp_desc.inf.model

        if isinstance(model_desc, ed.MDN_Descriptor):

            import ml.models.mdns as mdns

            return mdns.MDN(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_hiddens=model_desc.n_hiddens,
                act_fun=model_desc.act_fun,
                n_components=model_desc.n_comps,
                rng=rng
            )

        elif isinstance(model_desc, ed.MAF_Descriptor):

            import ml.models.mafs as mafs

            return mafs.ConditionalMaskedAutoregressiveFlow(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_hiddens=model_desc.n_hiddens,
                act_fun=model_desc.act_fun,
                n_mades=model_desc.n_comps,
                mode='random',
                rng=rng
            )

        else:
            raise TypeError('unknown model descriptor')
