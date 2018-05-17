import os
import numpy as np

import ml.models.neural_nets as nn
import ml.trainers as trainers
import ml.loss_functions as lf

import simulators.lotka_volterra as sim

import misc

import util.io
import util.plot


dir = os.path.join(misc.get_root(), 'results', 'lotka_volterra', 'other', 'failed_sims_model')


def gen_data(n_data=100000, rng=np.random):
    """
    Generates training data to fit the model.
    :param n_data: number of datapoints
    :param rng: random number generator
    """

    res_file = os.path.join(dir, 'data')

    if os.path.exists(res_file + '.pkl'):
        ps, ys = util.io.load(res_file)

    else:
        prior = sim.Prior()
        model = sim.Model()

        ps = prior.gen(n_data, rng=rng)
        xs = model.sim(ps, rng=rng)
        ys = np.array([0.0 if x is None else 1.0 for x in xs])

        util.io.save((ps, ys), res_file)

    return ps, ys


def create_net(rng=np.random):
    """
    Creates a network with logistic output.
    """

    n_inputs = sim.Prior().n_dims

    net = nn.FeedforwardNet(n_inputs)
    net.addLayer(100, 'relu', rng=rng)
    net.addLayer(100, 'relu', rng=rng)
    net.addLayer(1, 'logistic', rng=rng)

    return net


def train(net, ps, ys, val_frac=0.05, rng=np.random):
    """
    Trains a network to predict whether a simulation will fail.
    :param net: network to train
    :param ps: training inputs (parameters from prior)
    :param ys: training labels (whether simulation failed)
    :param val_frac: fraction of data to use for validation.
    :param rng: random number generator
    :return: trained net
    """

    ps = np.asarray(ps, np.float32)
    ys = np.asarray(ys, np.float32)

    n_data = ps.shape[0]
    assert ys.shape[0] == n_data, 'wrong sizes'

    # shuffle data, so that training and validation sets come from the same distribution
    idx = rng.permutation(n_data)
    ps = ps[idx]
    ys = ys[idx]

    # split data into training and validation sets
    n_trn = int(n_data - val_frac * n_data)
    xs_trn, xs_val = ps[:n_trn], ps[n_trn:]
    ys_trn, ys_val = ys[:n_trn], ys[n_trn:]

    trn_target, trn_loss = lf.CrossEntropy(net.output)

    trainer = trainers.SGD(
        model=net,
        trn_data=[xs_trn, ys_trn],
        trn_loss=trn_loss,
        trn_target=trn_target,
        val_data=[xs_val, ys_val],
        val_loss=trn_loss,
        val_target=trn_target
    )
    trainer.train(
        minibatch=100,
        patience=30,
        monitor_every=1
    )

    return net


def plot_net(n_samples=10000):
    """
    Plots network predictions and actual samples for which simulations didn't fail.
    """

    ps = sim.Prior().gen(n_samples)
    net = util.io.load(os.path.join(dir, 'model'))
    ys = net.eval(ps)[:, 0]
    util.plot.plot_hist_marginals(ps, weights=ys, lims=sim.get_disp_lims()).suptitle('learned')

    ps_gt, _ = sim.SimsLoader().load(n_samples)
    util.plot.plot_hist_marginals(ps_gt, lims=sim.get_disp_lims()).suptitle('true')

    util.plot.plt.show()


def main():

    print 'Learning failed sims model for lotka-volterra'

    rng = np.random.RandomState(42)

    ps, ys = gen_data()
    net = create_net(rng=rng)
    net = train(net, ps, ys, rng=rng)

    util.io.save(net, os.path.join(dir, 'model'))

    print 'ALL DONE'


if __name__ == '__main__':
    main()
