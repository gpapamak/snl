from __future__ import division

import numpy as np
import theano
import theano.tensor as tt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import matplotlib.pyplot as plt

import util.math
import util.ml
import util.plot

dtype = theano.config.floatX


class FeedforwardNet:
    """
    Implements a feedforward neural network.
    Supports various types of layers and loss functions.
    """

    def __init__(self, n_inputs, input=None):
        """
        Constructs a net with a given number of inputs and no layers.
        """

        assert util.math.isposint(n_inputs), 'Number of inputs must be a positive integer.'

        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.n_units = [n_inputs]
        self.n_layers = 0
        self.n_params = 0

        self.Ws = []
        self.bs = []
        self.hs = [tt.matrix('x') if input is None else input]
        self.parms = self.Ws + self.bs
        self.input = self.hs[0]
        self.output = self.hs[-1]

        # theano function
        self.eval_f = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_f = None

    def addLayer(self, n_units, type, rng=np.random):
        """
        Adds a new layer to the network,
        :param n_units: number of units in the layer
        :param type: a string specification of the activation function
        """

        # check number of units
        assert util.math.isposint(n_units), 'Number of units must be a positive integer.'

        # choose activation function
        actfun = util.ml.select_theano_act_function(type, dtype)

        n_prev_units = self.n_outputs
        self.n_outputs = n_units
        self.n_units.append(n_units)
        self.n_layers += 1
        self.n_params += (n_prev_units + 1) * n_units

        W = theano.shared((rng.randn(n_prev_units, n_units) / np.sqrt(n_prev_units + 1)).astype(dtype), name='W' + str(self.n_layers), borrow=True)
        b = theano.shared(np.zeros(n_units, dtype=dtype), name='b' + str(self.n_layers), borrow=True)
        h = actfun(tt.dot(self.hs[-1], W) + b)
        h.name = 'h' + str(self.n_layers)

        self.Ws.append(W)
        self.bs.append(b)
        self.hs.append(h)
        self.parms = self.Ws + self.bs
        self.output = self.hs[-1]

        self.reset_theano_functions()

    def removeLayer(self):
        """
        Removes a layer from the network.
        """

        assert self.n_layers > 0, 'There is no layer to remove.'

        n_params_to_rem = self.n_outputs * (self.n_units[-2] + 1)
        self.n_outputs = self.n_units[-2]
        self.n_units.pop()
        self.n_layers -= 1
        self.n_params -= n_params_to_rem

        self.Ws.pop()
        self.bs.pop()
        self.hs.pop()
        self.parms = self.Ws + self.bs
        self.output = self.hs[-1]

        self.reset_theano_functions()

    def eval(self, x):
        """
        Evaluate net at locations in x.
        """

        # compile theano computation graph, if haven't already done so
        if self.eval_f is None:
            self.eval_f = theano.function(
                inputs=[self.hs[0]],
                outputs=self.hs[-1]
            )

        x = np.asarray(x, dtype=dtype)

        return self.eval_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_f(x)

    def printInfo(self):
        """
        Prints some useful info about the net.
        """

        print 'Number of inputs  =', self.n_inputs
        print 'Number of outputs =', self.n_outputs
        print 'Number of units   =', self.n_units
        print 'Number of layers  =', self.n_layers
        print 'Number of params  =', self.n_params
        print 'Data type =', dtype

    def visualize_weights(self, layer, imsize, layout):
        """
        Displays the weights of a specified layer as images.
        :param layer: the layer whose weights to display
        :param imsize: the image size
        :param layout: number of rows and columns for each page
        :return: none
        """

        util.plot.disp_imdata(self.Ws[layer].get_value().T, imsize, layout)
        plt.show(block=False)

    def visualize_activations(self, x, layers=None):
        """
        Visualizes the activations of specified layers caused by a given data minibatch.
        :param x: a minibatch of data
        :param layers: list of layers to visualize activations of; defaults to the whole net except the input layer
        :return: none
        """

        if layers is None:
            layers = xrange(self.n_layers)

        forwprop = theano.function(
            inputs=[self.hs[0]],
            outputs=self.hs[1:]
        )
        hs = forwprop(x.astype(dtype))

        for l in layers:

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(hs[l], cmap='gray', interpolation='none')
            ax.set_title('Layer ' + str(l))
            ax.set_xlabel('layer units')
            ax.set_ylabel('data points')

        plt.show(block=False)

    def param_hist(self, layers=None):
        """
        Displays a histogram of weights and biases for specified layers.
        :param layers: list of layers to show histograms for; defaults to the whole net
        :return: none
        """

        if layers is None:
            layers = xrange(self.n_layers)

        for l in layers:

            fig, (ax1, ax2) = plt.subplots(1, 2)

            nbins = int(np.sqrt(self.Ws[l].get_value().size))
            ax1.hist(self.Ws[l].get_value().flatten(), nbins, normed=True)
            ax1.set_title('weights, layer ' + str(l))

            nbins = int(np.sqrt(self.bs[l].get_value().size))
            ax2.hist(self.bs[l].get_value(), nbins, normed=True)
            ax2.set_title('biases, layer ' + str(l))

        plt.show(block=False)


class FeedforwardNet_SVI:
    """
    Implements a feedforward neural network trained using stochastic variational inference.
    Supports various types of layers and loss functions.
    """

    def __init__(self, n_inputs):
        """
        Constructs a net with a given number of inputs and no layers.
        """

        assert util.math.isposint(n_inputs), 'Number of inputs must be a positive integer.'

        self.n_inputs = n_inputs
        self.n_outputs = n_inputs
        self.n_units = [n_inputs]
        self.n_layers = 0
        self.n_params = 0

        self.mWs = []
        self.mbs = []
        self.sWs = []
        self.sbs = []
        self.uas = []
        self.mas = []
        self.zas = []
        self.hs = [tt.matrix('x')]

        self.mps = self.mWs + self.mbs
        self.sps = self.sWs + self.sbs
        self.parms = self.mps + self.sps
        self.input = self.hs[0]
        self.output = self.hs[-1]

        self.srng = RandomStreams()

        # theano functions
        self.eval_f = None
        self.eval_f_rand = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_f = None
        self.eval_f_rand = None

    def addLayer(self, n_units, type, rng=np.random):
        """
        Adds a new layer to the network,
        :param n_units: number of units in the layer
        :param type: a string specification of the activation function
        """

        # check number of units
        assert util.math.isposint(n_units), 'Number of units must be a positive integer.'

        # choose activation function
        actfun = util.ml.select_theano_act_function(type, dtype)

        n_prev_units = self.n_outputs
        self.n_outputs = n_units
        self.n_units.append(n_units)
        self.n_layers += 1
        self.n_params += 2 * (n_prev_units + 1) * n_units

        mW = theano.shared((rng.randn(n_prev_units, n_units) / np.sqrt(n_prev_units + 1)).astype(dtype), name='mW' + str(self.n_layers), borrow=True)
        mb = theano.shared(np.zeros(n_units, dtype=dtype), name='mb' + str(self.n_layers), borrow=True)
        sW = theano.shared(-5.0 * np.ones([n_prev_units, n_units], dtype=dtype), name='sW' + str(self.n_layers), borrow=True)
        sb = theano.shared(-5.0 * np.ones(n_units, dtype=dtype), name='sb' + str(self.n_layers), borrow=True)
        ua = self.srng.normal((self.hs[-1].shape[0], n_units), dtype=dtype)
        ma = tt.dot(self.hs[-1], mW) + mb
        sa = tt.dot(self.hs[-1]**2, tt.exp(2*sW)) + tt.exp(2*sb)
        za = tt.sqrt(sa) * ua + ma
        h = actfun(za)
        h.name = 'h' + str(self.n_layers)

        self.mWs.append(mW)
        self.mbs.append(mb)
        self.sWs.append(sW)
        self.sbs.append(sb)
        self.uas.append(ua)
        self.mas.append(ma)
        self.zas.append(za)
        self.hs.append(h)

        self.mps = self.mWs + self.mbs
        self.sps = self.sWs + self.sbs
        self.parms = self.mps + self.sps
        self.output = self.hs[-1]

        self.reset_theano_functions()

    def removeLayer(self):
        """
        Removes a layer from the network.
        """

        assert self.n_layers > 0, 'There is no layer to remove.'

        n_params_to_rem = 2 * self.n_outputs * (self.n_units[-2] + 1)
        self.n_outputs = self.n_units[-2]
        self.n_units.pop()
        self.n_layers -= 1
        self.n_params -= n_params_to_rem

        self.mWs.pop()
        self.mbs.pop()
        self.sWs.pop()
        self.sbs.pop()
        self.uas.pop()
        self.mas.pop()
        self.zas.pop()
        self.hs.pop()

        self.mps = self.mWs + self.mbs
        self.sps = self.sWs + self.sbs
        self.parms = self.mps + self.sps
        self.output = self.hs[-1]

        self.reset_theano_functions()

    def eval(self, x, rand=False):
        """
        Evaluate net at locations in x.
        """

        x = np.asarray(x, dtype=dtype)

        if rand:

            # compile theano computation graph, if haven't already done so
            if self.eval_f_rand is None:

                n_data = tt.iscalar('n_data')
                uas = [tt.tile(self.srng.normal((n_units,), dtype=dtype), [n_data, 1]) for n_units in self.n_units[1:]]

                self.eval_f_rand = theano.function(
                    inputs=[self.hs[0], n_data],
                    outputs=self.hs[-1],
                    givens=zip(self.uas, uas)
                )

            return self.eval_f_rand(x[np.newaxis, :], 1)[0] if x.ndim == 1 else self.eval_f_rand(x, x.shape[0])

        else:

            # compile theano computation graph, if haven't already done so
            if self.eval_f is None:
                self.eval_f = theano.function(
                    inputs=[self.hs[0]],
                    outputs=self.hs[-1],
                    givens=zip(self.zas, self.mas)
                )

            return self.eval_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_f(x)

    def printInfo(self):
        """
        Prints some useful info about the net.
        """

        print 'Number of inputs  =', self.n_inputs
        print 'Number of outputs =', self.n_outputs
        print 'Number of units   =', self.n_units
        print 'Number of layers  =', self.n_layers
        print 'Number of params  =', self.n_params
        print 'Data type =', dtype

    def visualize_weights(self, layer, imsize, layout):
        """
        Displays the weights of a specified layer as images.
        :param layer: the layer whose weights to display
        :param imsize: the image size
        :param layout: number of rows and columns for each page
        :return: none
        """

        util.plot.disp_imdata(self.mWs[layer].get_value().T, imsize, layout)
        plt.show(block=False)

    def visualize_activations(self, x, layers=None):
        """
        Visualizes the activations of specified layers caused by a given data minibatch.
        :param x: a minibatch of data
        :param layers: list of layers to visualize activations of; defaults to the whole net except the input layer
        :return: none
        """

        if layers is None:
            layers = xrange(self.n_layers)

        forwprop = theano.function(
            inputs=[self.hs[0]],
            outputs=self.hs[1:]
        )
        hs = forwprop(x.astype(dtype))

        for l in layers:

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(hs[l], cmap='gray', interpolation='none')
            ax.set_title('Layer ' + str(l))
            ax.set_xlabel('layer units')
            ax.set_ylabel('data points')

        plt.show(block=False)

    def param_hist(self, layers=None):
        """
        Displays a histogram of weights and biases for specified layers.
        :param layers: list of layers to show histograms for; defaults to the whole net
        :return: none
        """

        if layers is None:
            layers = xrange(self.n_layers)

        for l in layers:

            fig, axs = plt.subplots(2, 2)

            nbins = int(np.sqrt(self.mWs[l].get_value().size))
            axs[0, 0].hist(self.mWs[l].get_value().flatten(), nbins, normed=True)
            axs[0, 0].set_title('weight means, layer ' + str(l))
            axs[1, 0].hist(self.sWs[l].get_value().flatten(), nbins, normed=True)
            axs[1, 0].set_title('weight log stds, layer ' + str(l))

            nbins = int(np.sqrt(self.mbs[l].get_value().size))
            axs[0, 1].hist(self.mbs[l].get_value(), nbins, normed=True)
            axs[0, 1].set_title('bias means, layer ' + str(l))
            axs[1, 1].hist(self.sbs[l].get_value(), nbins, normed=True)
            axs[1, 1].set_title('bias log stds, layer ' + str(l))

        plt.show(block=False)
