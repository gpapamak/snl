from itertools import izip

import numpy as np
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import util.misc
import util.ml
import util.math

dtype = theano.config.floatX


# ========================================================================
# THESE ARE JUST TESTING ROUTINES

def calc_connectivity(Ms, Mmp):

    C = 1

    for M in Ms + [Mmp]:
        C = np.dot(C, M.get_value())

    return C


def test_connectivity(C):

    N1, N2 = C.shape
    all = 0
    gaps = 0

    for i in xrange(N1):
        for j in xrange(i+1, N2):
            all += 1
            if C[i, j] == 0:
                gaps += 1

    not_connected = float(gaps) / all
    n_connections = int(np.sum(C))

    return not_connected, n_connections


def test_autoregressive_masks(n_inputs, n_hiddens, mode):

    degrees = create_degrees(n_inputs, n_hiddens, 'sequential', mode)
    Ms, Mmp = create_masks(degrees)
    C = calc_connectivity(Ms, Mmp)
    not_connected, n_connections = test_connectivity(C)

    print 'not connected = {0:.2%}'.format(not_connected)
    print '# connections = {0}'.format(n_connections)

    fig, ax = plt.subplots(1, 1)
    fig.colorbar(ax.matshow(C))
    plt.show()

# ========================================================================


def create_degrees(n_inputs, n_hiddens, input_order, mode, rng=np.random):
    """
    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than d.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param input_order: the order of the inputs; can be 'random', 'sequential', or an array of an explicit order
    :param mode: the strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
    :param rng: random number generator
    :return: list of degrees
    """

    degrees = []

    # create degrees for inputs
    if isinstance(input_order, str):

        if input_order == 'random':
            degrees_0 = np.arange(1, n_inputs + 1)
            rng.shuffle(degrees_0)

        elif input_order == 'sequential':
            degrees_0 = np.arange(1, n_inputs + 1)

        else:
            raise ValueError('invalid input order')

    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hiddens
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = rng.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)

    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % max(1, n_inputs - 1) + min(1, n_inputs - 1)
            degrees.append(degrees_l)

    else:
        raise ValueError('invalid mode')

    return degrees


def create_masks(degrees):
    """
    Creates the binary masks that make the connectivity autoregressive.
    :param degrees: a list of degrees for every layer
    :return: list of all masks, as theano shared variables
    """

    Ms = []

    for l, (d0, d1) in enumerate(izip(degrees[:-1], degrees[1:])):
        M = d0[:, np.newaxis] <= d1
        M = theano.shared(M.astype(dtype), name='M' + str(l+1), borrow=True)
        Ms.append(M)

    Mmp = degrees[-1][:, np.newaxis] < degrees[0]
    Mmp = theano.shared(Mmp.astype(dtype), name='Mmp', borrow=True)

    return Ms, Mmp


def create_weights(n_inputs, n_hiddens, n_comps, rng=np.random):
    """
    Creates all learnable weight matrices and bias vectors.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :param rng: random number generator
    :return: weights and biases, as theano shared variables
    """

    Ws = []
    bs = []

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for l, (N0, N1) in enumerate(izip(n_units[:-1], n_units[1:])):
        W = theano.shared((rng.randn(N0, N1) / np.sqrt(N0 + 1)).astype(dtype), name='W' + str(l+1), borrow=True)
        b = theano.shared(np.zeros(N1, dtype=dtype), name='b' + str(l+1), borrow=True)
        Ws.append(W)
        bs.append(b)

    if n_comps is None:

        Wm = theano.shared((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='Wm', borrow=True)
        Wp = theano.shared((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='Wp', borrow=True)
        bm = theano.shared(np.zeros(n_inputs, dtype=dtype), name='bm', borrow=True)
        bp = theano.shared(np.zeros(n_inputs, dtype=dtype), name='bp', borrow=True)

        return Ws, bs, Wm, bm, Wp, bp

    else:

        Wm = theano.shared((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='Wm', borrow=True)
        Wp = theano.shared((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='Wp', borrow=True)
        Wa = theano.shared((rng.randn(n_units[-1], n_inputs, n_comps) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='Wa', borrow=True)
        bm = theano.shared(rng.randn(n_inputs, n_comps).astype(dtype), name='bm', borrow=True)
        bp = theano.shared(rng.randn(n_inputs, n_comps).astype(dtype), name='bp', borrow=True)
        ba = theano.shared(rng.randn(n_inputs, n_comps).astype(dtype), name='ba', borrow=True)

        return Ws, bs, Wm, bm, Wp, bp, Wa, ba


def create_weights_conditional(n_inputs, n_outputs, n_hiddens, n_comps, rng):
    """
    Creates all learnable weight matrices and bias vectors for a conditional made.
    :param n_inputs: the number of (conditional) inputs
    :param n_outputs: the number of outputs
    :param n_hiddens: a list with the number of hidden units
    :param n_comps: number of gaussian components
    :param rng: random number generator
    :return: weights and biases, as theano shared variables
    """

    Wx = theano.shared((rng.randn(n_inputs, n_hiddens[0]) / np.sqrt(n_inputs + 1)).astype(dtype), name='Wx', borrow=True)

    return (Wx,) + create_weights(n_outputs, n_hiddens, n_comps, rng)


def create_weights_SVI(n_inputs, n_hiddens, rng):
    """
    Creates all learnable weight matrices and bias vectors for the SVI version of made.
    :param n_inputs: the number of inputs
    :param n_hiddens: a list with the number of hidden units
    :param rng: random number generator
    :return: weights and biases, as theano shared variables
    """

    # feedforward weights
    mWs = []
    mbs = []
    sWs = []
    sbs = []

    n_units = np.concatenate(([n_inputs], n_hiddens))

    for l, (N0, N1) in enumerate(izip(n_units[:-1], n_units[1:])):
        mW = theano.shared((rng.randn(N0, N1) / np.sqrt(N0 + 1)).astype(dtype), name='mW' + str(l+1), borrow=True)
        mb = theano.shared(np.zeros(N1, dtype=dtype), name='mb' + str(l+1), borrow=True)
        sW = theano.shared(-5.0 * np.ones([N0, N1], dtype=dtype), name='sW' + str(l+1), borrow=True)
        sb = theano.shared(-5.0 * np.ones(N1, dtype=dtype), name='sb' + str(l+1), borrow=True)
        mWs.append(mW)
        mbs.append(mb)
        sWs.append(sW)
        sbs.append(sb)

    # weights from last layer to means
    mWm = theano.shared((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='mWm', borrow=True)
    mbm = theano.shared(np.zeros(n_inputs, dtype=dtype), name='mbm', borrow=True)
    sWm = theano.shared(-5.0 * np.ones([n_units[-1], n_inputs], dtype=dtype), name='sWm', borrow=True)
    sbm = theano.shared(-5.0 * np.ones(n_inputs, dtype=dtype), name='sbm', borrow=True)

    # weights from last layer to log precisions
    mWp = theano.shared((rng.randn(n_units[-1], n_inputs) / np.sqrt(n_units[-1] + 1)).astype(dtype), name='mWp', borrow=True)
    mbp = theano.shared(np.zeros(n_inputs, dtype=dtype), name='mbp', borrow=True)
    sWp = theano.shared(-5.0 * np.ones([n_units[-1], n_inputs], dtype=dtype), name='sWp', borrow=True)
    sbp = theano.shared(-5.0 * np.ones(n_inputs, dtype=dtype), name='sbp', borrow=True)

    return mWs, mbs, sWs, sbs, mWm, mbm, sWm, sbm, mWp, mbp, sWp, sbp


class GaussianMade:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component.
    Reference: Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML, 2015.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, input_order='sequential', mode='sequential', input=None, rng=np.random):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_inputs, n_hiddens, input_order, mode, rng)
        Ms, Mmp = create_masks(degrees)
        Ws, bs, Wm, bm, Wp, bp = create_weights(n_inputs, n_hiddens, None, rng)
        self.parms = Ws + bs + [Wm, bm, Wp, bp]
        self.input_order = degrees[0]

        # activation function
        f = util.ml.select_theano_act_function(act_fun, dtype)

        # input matrix
        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        h = self.input

        # feedforward propagation
        for l, (M, W, b) in enumerate(izip(Ms, Ws, bs)):
            h = f(tt.dot(h, M * W) + b)
            h.name = 'h' + str(l + 1)

        # output means
        self.m = tt.dot(h, Mmp * Wm) + bm
        self.m.name = 'm'

        # output log precisions
        self.logp = tt.dot(h, Mmp * Wp) + bp
        self.logp.name = 'logp'

        # random numbers driving made
        self.u = tt.exp(0.5 * self.logp) * (self.input - self.m)

        # log likelihoods
        self.L = -0.5 * (n_inputs * np.log(2 * np.pi) + tt.sum(self.u ** 2 - self.logp, axis=1))
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_us_f = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_us_f = None

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input],
                outputs=self.L
            )

        x = np.asarray(x, dtype=dtype)
        lprob = self.eval_lprob_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_lprob_f(x)

        return lprob if log else np.exp(lprob)

    def eval_comps(self, x):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param x: rows are input locations
        :return: means and log precisions
        """

        # compile theano function, if haven't already done so
        if self.eval_comps_f is None:
            self.eval_comps_f = theano.function(
                inputs=[self.input],
                outputs=[self.m, self.logp]
            )

        x = np.asarray(x, dtype=dtype)

        if x.ndim == 1:
            m, logp = self.eval_comps_f(x[np.newaxis, :])
            return m[0], logp[0]
        else:
            return self.eval_comps_f(x)

    def grad_log_p(self, x):
        """
        Evaluate the gradient of the log probability wrt the input.
        :param x: rows are input locations
        :return: gradient d/dx log p(x)
        """

        # compile theano function, if haven't already done so
        if getattr(self, 'eval_grad_f', None) is None:
            self.eval_grad_f = theano.function(
                inputs=[self.input],
                outputs=tt.grad(tt.sum(self.L), self.input)
            )

        x = np.asarray(x, dtype=dtype)
        grad = self.eval_grad_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_grad_f(x)

        return grad

    def gen(self, n_samples=None, u=None, rng=np.random):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples, 1 if None
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        if n_samples is None:
            return self.gen(1, u if u is None else u[np.newaxis, :], rng)[0]

        x = np.zeros([n_samples, self.n_inputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        for i in xrange(1, self.n_inputs + 1):
            m, logp = self.eval_comps(x)
            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return x

    def calc_random_numbers(self, x):
        """
        Givan a dataset, calculate the random numbers made uses internally to generate the dataset.
        :param x: numpy array, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input],
                outputs=self.u
            )

        x = np.asarray(x, dtype=dtype)

        return self.eval_us_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_us_f(x)


class MixtureOfGaussiansMade:
    """
    Implements a Made, where each conditional probability is modelled by a mixture of gaussians.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, n_comps, input_order='sequential', mode='sequential', input=None, rng=np.random):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_comps: number of gaussian components
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_comps = n_comps
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_inputs, n_hiddens, input_order, mode, rng)
        Ms, Mmp = create_masks(degrees)
        Mmp_broadcast = Mmp.dimshuffle([0, 1, 'x'])
        Ws, bs, Wm, bm, Wp, bp, Wa, ba = create_weights(n_inputs, n_hiddens, n_comps, rng)
        self.parms = Ws + bs + [Wm, bm, Wp, bp, Wa, ba]
        self.input_order = degrees[0]

        # activation function
        f = util.ml.select_theano_act_function(act_fun, dtype)

        # input matrix
        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        h = self.input

        # feedforward propagation
        for l, (M, W, b) in enumerate(izip(Ms, Ws, bs)):
            h = f(tt.dot(h, M * W) + b)
            h.name = 'h' + str(l + 1)

        # output means
        self.m = tt.tensordot(h, Mmp_broadcast * Wm, axes=[1, 0]) + bm
        self.m.name = 'm'

        # output log precisions
        self.logp = tt.tensordot(h, Mmp_broadcast * Wp, axes=[1, 0]) + bp
        self.logp.name = 'logp'

        # output mixing coefficients
        self.loga = tt.tensordot(h, Mmp_broadcast * Wa, axes=[1, 0]) + ba
        self.loga -= tt.log(tt.sum(tt.exp(self.loga), axis=2, keepdims=True))
        self.loga.name = 'loga'

        # random numbers driving made
        self.u = tt.exp(0.5 * self.logp) * (self.input.dimshuffle([0, 1, 'x']) - self.m)

        # log likelihoods
        self.L = tt.log(tt.sum(tt.exp(self.loga - 0.5 * self.u ** 2 + 0.5 * self.logp), axis=2))
        self.L = -0.5 * n_inputs * np.log(2 * np.pi) + tt.sum(self.L, axis=1)
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_us_f = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_us_f = None

    def eval(self, x, log=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :return: list of log probabilities log p(x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input],
                outputs=self.L
            )

        x = np.asarray(x, dtype=dtype)
        lprob = self.eval_lprob_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_lprob_f(x)

        return lprob if log else np.exp(lprob)

    def eval_comps(self, x):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param x: rows are input locations
        :return: means, log precisions and log mixing coefficients
        """

        # compile theano function, if haven't already done so
        if self.eval_comps_f is None:
            self.eval_comps_f = theano.function(
                inputs=[self.input],
                outputs=[self.m, self.logp, self.loga]
            )

        x = np.asarray(x, dtype=dtype)

        if x.ndim == 1:
            m, logp, loga = self.eval_comps_f(x[np.newaxis, :])
            return m[0], logp[0], loga[0]
        else:
            return self.eval_comps_f(x)

    def grad_log_p(self, x):
        """
        Evaluate the gradient of the log probability wrt the input.
        :param x: rows are input locations
        :return: gradient d/dx log p(x)
        """

        # compile theano function, if haven't already done so
        if getattr(self, 'eval_grad_f', None) is None:
            self.eval_grad_f = theano.function(
                inputs=[self.input],
                outputs=tt.grad(tt.sum(self.L), self.input)
            )

        x = np.asarray(x, dtype=dtype)
        grad = self.eval_grad_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_grad_f(x)

        return grad

    def gen(self, n_samples=None, u=None, rng=np.random):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples, 1 if None
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        if n_samples is None:
            return self.gen(1, u if u is None else u[np.newaxis, :], rng)[0]

        x = np.zeros([n_samples, self.n_inputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        for i in xrange(1, self.n_inputs + 1):
            m, logp, loga = self.eval_comps(x)
            idx = np.argwhere(self.input_order == i)[0, 0]
            for n in xrange(n_samples):
                z = util.math.discrete_sample(np.exp(loga[n, idx]), rng=rng)
                x[n, idx] = m[n, idx, z] + np.exp(np.minimum(-0.5 * logp[n, idx, z], 10.0)) * u[n, idx]

        return x

    def calc_random_numbers(self, x):
        """
        Givan a dataset, calculate the random numbers made uses internally to generate the dataset.
        :param x: numpy array, rows are datapoints
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input],
                outputs=self.u
            )

        x = np.asarray(x, dtype=dtype)

        return self.eval_us_f(x[np.newaxis, :])[0] if x.ndim == 1 else self.eval_us_f(x)


class ConditionalGaussianMade:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component. The made has
    inputs which is always conditioned on, and whose probability it doesn't model.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, output_order='sequential', mode='sequential', input=None, output=None, rng=np.random):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param output_order: order of outputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        :param output: theano variable to serve as output; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_outputs, n_hiddens, output_order, mode, rng)
        Ms, Mmp = create_masks(degrees)
        Wx, Ws, bs, Wm, bm, Wp, bp = create_weights_conditional(n_inputs, n_outputs, n_hiddens, None, rng)
        self.parms = [Wx] + Ws + bs + [Wm, bm, Wp, bp]
        self.output_order = degrees[0]

        # activation function
        f = util.ml.select_theano_act_function(act_fun, dtype)

        # input matrices
        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.y = tt.matrix('y', dtype=dtype) if output is None else output

        # feedforward propagation
        h = f(tt.dot(self.input, Wx) + tt.dot(self.y, Ms[0] * Ws[0]) + bs[0])
        h.name = 'h1'
        for l, (M, W, b) in enumerate(izip(Ms[1:], Ws[1:], bs[1:])):
            h = f(tt.dot(h, M * W) + b)
            h.name = 'h' + str(l + 2)

        # output means
        self.m = tt.dot(h, Mmp * Wm) + bm
        self.m.name = 'm'

        # output log precisions
        self.logp = tt.dot(h, Mmp * Wp) + bp
        self.logp.name = 'logp'

        # random numbers driving made
        self.u = tt.exp(0.5 * self.logp) * (self.y - self.m)

        # log likelihoods
        self.L = -0.5 * (n_outputs * np.log(2 * np.pi) + tt.sum(self.u ** 2 - self.logp, axis=1))
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_score_f = None
        self.eval_us_f = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_score_f = None
        self.eval_us_f = None

    def eval(self, xy, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.L
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        lprob = self.eval_lprob_f(x, y)
        lprob = lprob[0] if one_datapoint else lprob

        return lprob if log else np.exp(lprob)

    def eval_comps(self, xy):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: means and log precisions
        """

        # compile theano function, if haven't already done so
        if self.eval_comps_f is None:
            self.eval_comps_f = theano.function(
                inputs=[self.input, self.y],
                outputs=[self.m, self.logp]
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        if one_datapoint:
            m, logp = self.eval_comps_f(x, y)
            return m[0], logp[0]
        else:
            return self.eval_comps_f(x, y)

    def grad_log_p(self, xy):
        """
        Evaluate the gradient of the log probability wrt the output, for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: gradient d/dy log p(y|x)
        """

        # compile theano function, if haven't already done so
        if getattr(self, 'eval_grad_f', None) is None:
            self.eval_grad_f = theano.function(
                inputs=[self.input, self.y],
                outputs=tt.grad(tt.sum(self.L), self.y)
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        grads = self.eval_grad_f(x, y)
        grads = grads[0] if one_datapoint else grads

        return grads

    def score(self, xy):
        """
        Evaluate the gradient of the log probability wrt the input, for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: gradient d/dx log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_score_f is None:
            self.eval_score_f = theano.function(
                inputs=[self.input, self.y],
                outputs=tt.grad(tt.sum(self.L), self.input)
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        grads = self.eval_score_f(x, y)
        grads = grads[0] if one_datapoint else grads

        return grads

    def gen(self, x, n_samples=None, u=None, rng=np.random):
        """
        Generate samples from made conditioned on x. Requires as many evaluations as number of outputs.
        :param x: input vector
        :param n_samples: number of samples, 1 if None
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        if n_samples is None:
            return self.gen(x, 1, u if u is None else u[np.newaxis, :], rng)[0]

        y = np.zeros([n_samples, self.n_outputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u

        xy = (np.tile(x, [n_samples, 1]), y)

        for i in xrange(1, self.n_outputs + 1):
            m, logp = self.eval_comps(xy)
            idx = np.argwhere(self.output_order == i)[0, 0]
            y[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers made uses internally to generate the dataset.
        :param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.u
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
        u = self.eval_us_f(x, y)

        return u[0] if one_datapoint else u


class ConditionalMixtureOfGaussiansMade:
    """
    Implements a conditional Made, where each conditional probability is modelled by a mixture of gaussians.
    """

    def __init__(self, n_inputs, n_outputs, n_hiddens, act_fun, n_comps, output_order='sequential', mode='sequential', input=None, output=None, rng=np.random):
        """
        Constructor.
        :param n_inputs: number of (conditional) inputs
        :param n_outputs: number of outputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param n_comps: number of gaussian components
        :param output_order: order of outputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        :param output: theano variable to serve as output; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.n_comps = n_comps
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_outputs, n_hiddens, output_order, mode, rng)
        Ms, Mmp = create_masks(degrees)
        Mmp_broadcast = Mmp.dimshuffle([0, 1, 'x'])
        Wx, Ws, bs, Wm, bm, Wp, bp, Wa, ba = create_weights_conditional(n_inputs, n_outputs, n_hiddens, n_comps, rng)
        self.parms = [Wx] + Ws + bs + [Wm, bm, Wp, bp, Wa, ba]
        self.output_order = degrees[0]

        # activation function
        f = util.ml.select_theano_act_function(act_fun, dtype)

        # input matrices
        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        self.y = tt.matrix('y', dtype=dtype) if output is None else output

        # feedforward propagation
        h = f(tt.dot(self.input, Wx) + tt.dot(self.y, Ms[0] * Ws[0]) + bs[0])
        h.name = 'h1'
        for l, (M, W, b) in enumerate(izip(Ms[1:], Ws[1:], bs[1:])):
            h = f(tt.dot(h, M * W) + b)
            h.name = 'h' + str(l + 2)

        # output means
        self.m = tt.tensordot(h, Mmp_broadcast * Wm, axes=[1, 0]) + bm
        self.m.name = 'm'

        # output log precisions
        self.logp = tt.tensordot(h, Mmp_broadcast * Wp, axes=[1, 0]) + bp
        self.logp.name = 'logp'

        # output mixing coefficients
        self.loga = tt.tensordot(h, Mmp_broadcast * Wa, axes=[1, 0]) + ba
        self.loga -= tt.log(tt.sum(tt.exp(self.loga), axis=2, keepdims=True))
        self.loga.name = 'loga'

        # random numbers driving made
        self.u = tt.exp(0.5 * self.logp) * (self.y.dimshuffle([0, 1, 'x']) - self.m)

        # log likelihoods
        self.L = tt.log(tt.sum(tt.exp(self.loga - 0.5 * self.u ** 2 + 0.5 * self.logp), axis=2))
        self.L = -0.5 * n_outputs * np.log(2 * np.pi) + tt.sum(self.L, axis=1)
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_score_f = None
        self.eval_us_f = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_grad_f = None
        self.eval_score_f = None
        self.eval_us_f = None

    def eval(self, xy, log=True):
        """
        Evaluate log probabilities for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :param log: whether to return probabilities in the log domain
        :return: log probabilities: log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_lprob_f is None:
            self.eval_lprob_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.L
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        lprob = self.eval_lprob_f(x, y)
        lprob = lprob[0] if one_datapoint else lprob

        return lprob if log else np.exp(lprob)

    def eval_comps(self, xy):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: means, log precisions and log mixing coefficients
        """

        # compile theano function, if haven't already done so
        if self.eval_comps_f is None:
            self.eval_comps_f = theano.function(
                inputs=[self.input, self.y],
                outputs=[self.m, self.logp, self.loga]
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
        comps = self.eval_comps_f(x, y)

        return map(lambda u: u[0], comps) if one_datapoint else comps

    def grad_log_p(self, xy):
        """
        Evaluate the gradient of the log probability wrt the output, for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: gradient d/dy log p(y|x)
        """

        # compile theano function, if haven't already done so
        if getattr(self, 'eval_grad_f', None) is None:
            self.eval_grad_f = theano.function(
                inputs=[self.input, self.y],
                outputs=tt.grad(tt.sum(self.L), self.y)
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        grads = self.eval_grad_f(x, y)
        grads = grads[0] if one_datapoint else grads

        return grads

    def score(self, xy):
        """
        Evaluate the gradient of the log probability wrt the input, for given input-output pairs.
        :param xy: a pair (x, y) where x rows are inputs and y rows are outputs
        :return: gradient d/dx log p(y|x)
        """

        # compile theano function, if haven't already done so
        if self.eval_score_f is None:
            self.eval_score_f = theano.function(
                inputs=[self.input, self.y],
                outputs=tt.grad(tt.sum(self.L), self.input)
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)

        grads = self.eval_score_f(x, y)
        grads = grads[0] if one_datapoint else grads

        return grads

    def gen(self, x, n_samples=None, u=None, rng=np.random):
        """
        Generate samples from made conditioned on x. Requires as many evaluations as number of outputs.
        :param x: input vector
        :param n_samples: number of samples, 1 if None
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        if n_samples is None:
            return self.gen(x, 1, u if u is None else u[np.newaxis, :], rng)[0]

        y = np.zeros([n_samples, self.n_outputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_outputs).astype(dtype) if u is None else u

        xy = (np.tile(x, [n_samples, 1]), y)

        for i in xrange(1, self.n_outputs + 1):
            m, logp, loga = self.eval_comps(xy)
            idx = np.argwhere(self.output_order == i)[0, 0]
            for n in xrange(n_samples):
                z = util.math.discrete_sample(np.exp(loga[n, idx]), rng=rng)
                y[n, idx] = m[n, idx, z] + np.exp(np.minimum(-0.5 * logp[n, idx, z], 10.0)) * u[n, idx]

        return y

    def calc_random_numbers(self, xy):
        """
        Givan a dataset, calculate the random numbers made uses internally to generate the dataset.
        :param xy: a pair (x, y) of numpy arrays, where x rows are inputs and y rows are outputs
        :return: numpy array, rows are corresponding random numbers
        """

        # compile theano function, if haven't already done so
        if self.eval_us_f is None:
            self.eval_us_f = theano.function(
                inputs=[self.input, self.y],
                outputs=self.u
            )

        x, y, one_datapoint = util.misc.prepare_cond_input(xy, dtype)
        u = self.eval_us_f(x, y)

        return u[0] if one_datapoint else u


class GaussianMade_SVI:
    """
    Implements a Made, where each conditional probability is modelled by a single gaussian component.
    This made is trained with stochastic variational inference, using the local reparameterization trick.
    References:
    Germain et al., "MADE: Masked Autoencoder for Distribution Estimation", ICML, 2015.
    Kingma et al., "Variational Dropout and the Local Reparameterization Trick", NIPS, 2015.
    """

    def __init__(self, n_inputs, n_hiddens, act_fun, input_order='sequential', mode='sequential', input=None, rng=np.random):
        """
        Constructor.
        :param n_inputs: number of inputs
        :param n_hiddens: list with number of hidden units for each hidden layer
        :param act_fun: name of activation function
        :param input_order: order of inputs
        :param mode: strategy for assigning degrees to hidden nodes: can be 'random' or 'sequential'
        :param input: theano variable to serve as input; if None, a new variable is created
        """

        # save input arguments
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.act_fun = act_fun
        self.mode = mode

        # create network's parameters
        degrees = create_degrees(n_inputs, n_hiddens, input_order, mode, rng)
        Ms, Mmp = create_masks(degrees)
        mWs, mbs, sWs, sbs, mWm, mbm, sWm, sbm, mWp, mbp, sWp, sbp = create_weights_SVI(n_inputs, n_hiddens, rng)
        self.mps = mWs + mbs + [mWm, mbm, mWp, mbp]
        self.sps = sWs + sbs + [sWm, sbm, sWp, sbp]
        self.parms = self.mps + self.sps
        self.input_order = degrees[0]
        self.srng = RandomStreams(rng.randint(2**30))

        # activation function
        f = util.ml.select_theano_act_function(act_fun, dtype)

        # input matrix
        self.input = tt.matrix('x', dtype=dtype) if input is None else input
        h = self.input
        uas = []

        # feedforward propagation
        for l, (M, mW, mb, sW, sb, N) in enumerate(izip(Ms, mWs, mbs, sWs, sbs, n_hiddens)):
            ma = tt.dot(h, M * mW) + mb
            sa = tt.dot(h**2, M * tt.exp(2*sW)) + tt.exp(2*sb)
            ua = self.srng.normal((h.shape[0], N), dtype=dtype)
            h = f(tt.sqrt(sa) * ua + ma)
            h.name = 'h' + str(l + 1)
            uas.append(ua)

        # output means
        mam = tt.dot(h, Mmp * mWm) + mbm
        sam = tt.dot(h**2, Mmp * tt.exp(2*sWm)) + tt.exp(2*sbm)
        uam = self.srng.normal((h.shape[0], n_inputs), dtype=dtype)
        self.m = tt.sqrt(sam) * uam + mam
        self.m.name = 'm'

        # output log precisions
        map = tt.dot(h, Mmp * mWp) + mbp
        sap = tt.dot(h**2, Mmp * tt.exp(2*sWp)) + tt.exp(2*sbp)
        uap = self.srng.normal((h.shape[0], n_inputs), dtype=dtype)
        self.logp = tt.sqrt(sap) * uap + map
        self.logp.name = 'logp'

        # random numbers driving made
        self.u = tt.exp(0.5 * self.logp) * (self.input - self.m)

        # log likelihoods
        self.L = -0.5 * (n_inputs * np.log(2 * np.pi) + tt.sum(self.u ** 2 - self.logp, axis=1))
        self.L.name = 'L'

        # train objective
        self.trn_loss = -tt.mean(self.L)
        self.trn_loss.name = 'trn_loss'

        # collect all noise variables
        self.all_us = uas + [uam, uap]

        # theano evaluation functions, will be compiled when first needed
        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_lprob_f_rand = None
        self.eval_comps_f_rand = None
        self.eval_lprob_f_rand_const = None
        self.eval_comps_f_rand_const = None

    def reset_theano_functions(self):
        """
        Resets theano functions, so that they are compiled again when needed.
        """

        self.eval_lprob_f = None
        self.eval_comps_f = None
        self.eval_lprob_f_rand = None
        self.eval_comps_f_rand = None
        self.eval_lprob_f_rand_const = None
        self.eval_comps_f_rand_const = None

    def _create_constant_noise_across_datapoints(self, n_data):
        """
        Helper function. Creates and returns new theano variables representing noise, where noise is the same across
        datapoints in the minibatch. Useful for binding the original noise variables in an evaluation function where
        randomness is required but same predictions are needed across minibatch.
        """

        uas = [tt.tile(self.srng.normal((N,), dtype=dtype), [n_data, 1]) for N in self.n_hiddens]
        uam = tt.tile(self.srng.normal((self.n_inputs,), dtype=dtype), [n_data, 1])
        uap = tt.tile(self.srng.normal((self.n_inputs,), dtype=dtype), [n_data, 1])

        return uas + [uam, uap]

    def _create_zero_noise(self, n_data):
        """
        Helper function. Creates and returns new theano variables representing zero noise. Useful for binding the
        original noise variables in an evaluation function where randomness is not required.
        """

        uas = [tt.zeros((n_data, N), dtype=dtype) for N in self.n_hiddens]
        uam = tt.zeros((n_data, self.n_inputs), dtype=dtype)
        uap = tt.zeros((n_data, self.n_inputs), dtype=dtype)

        return uas + [uam, uap]

    def eval(self, x, log=True, rand=False, const_noise=True):
        """
        Evaluate log probabilities for given inputs.
        :param x: data matrix where rows are inputs
        :param log: whether to return probabilities in the log domain
        :param rand: whether to inject randomness to the activations
        :param const_noise: whether the injected randomness is the same across datapoints
        :return: list of log probabilities log p(x)
        """

        x = np.asarray(x, dtype=dtype)
        one_datapoint = x.ndim == 1
        x = x[np.newaxis, :] if one_datapoint else x

        if rand:

            if const_noise:

                # compile theano function, if haven't already done so
                if self.eval_lprob_f_rand_const is None:

                    n_data = tt.iscalar('n_data')
                    all_us = self._create_constant_noise_across_datapoints(n_data)

                    self.eval_lprob_f_rand_const = theano.function(
                        inputs=[self.input, n_data],
                        outputs=self.L,
                        givens=zip(self.all_us, all_us)
                    )

                lprob = self.eval_lprob_f_rand_const(x, x.shape[0])

            else:

                # compile theano function, if haven't already done so
                if self.eval_lprob_f_rand is None:
                    self.eval_lprob_f_rand = theano.function(
                        inputs=[self.input],
                        outputs=self.L
                    )

                lprob = self.eval_lprob_f_rand(x)

        else:

            # compile theano function, if haven't already done so
            if self.eval_lprob_f is None:

                n_data = tt.iscalar('n_data')
                all_us = self._create_zero_noise(n_data)

                self.eval_lprob_f = theano.function(
                    inputs=[self.input, n_data],
                    outputs=self.L,
                    givens=zip(self.all_us, all_us)
                )

            lprob = self.eval_lprob_f(x, x.shape[0])

        lprob = lprob[0] if one_datapoint else lprob

        return lprob if log else np.exp(lprob)

    def eval_comps(self, x, rand=False, const_noise=False):
        """
        Evaluate the parameters of all gaussians at given input locations.
        :param x: rows are input locations
        :param rand: whether to inject randomness to the activations
        :param const_noise: whether the injected randomness is the same across datapoints
        :return: means and log precisions
        """

        x = np.asarray(x, dtype=dtype)
        one_datapoint = x.ndim == 1
        x = x[np.newaxis, :] if one_datapoint else x

        if rand:

            if const_noise:

                # compile theano function, if haven't already done so
                if self.eval_comps_f_rand_const is None:

                    n_data = tt.iscalar('n_data')
                    all_us = self._create_constant_noise_across_datapoints(n_data)

                    self.eval_comps_f_rand_const = theano.function(
                        inputs=[self.input, n_data],
                        outputs=[self.m, self.logp],
                        givens=zip(self.all_us, all_us)
                    )

                comps = self.eval_comps_f_rand_const(x, x.shape[0])

            else:

                # compile theano function, if haven't already done so
                if self.eval_comps_f_rand is None:
                    self.eval_comps_f_rand = theano.function(
                        inputs=[self.input],
                        outputs=[self.m, self.logp]
                    )

                comps = self.eval_comps_f_rand(x)

        else:

            # compile theano function, if haven't already done so
            if self.eval_comps_f is None:

                n_data = tt.iscalar('n_data')
                all_us = self._create_zero_noise(n_data)

                self.eval_comps_f = theano.function(
                    inputs=[self.input, n_data],
                    outputs=[self.m, self.logp],
                    givens=zip(self.all_us, all_us)
                )

            comps = self.eval_comps_f(x, x.shape[0])

        return map(lambda u: u[0], comps) if one_datapoint else comps

    def gen(self, n_samples=None, rand=False, const_noise=False, u=None, rng=np.random):
        """
        Generate samples from made. Requires as many evaluations as number of inputs.
        :param n_samples: number of samples, 1 if None
        :param rand: whether to inject randomness to the activations
        :param const_noise: whether the injected randomness is the same across samples
        :param u: random numbers to use in generating samples; if None, new random numbers are drawn
        :return: samples
        """

        if n_samples is None:
            return self.gen(1, rand, const_noise, u if u is None else u[np.newaxis, :], rng)[0]

        x = np.zeros([n_samples, self.n_inputs], dtype=dtype)
        u = rng.randn(n_samples, self.n_inputs).astype(dtype) if u is None else u

        # seed for theano random stream
        seed = rng.randint(2**30)

        for i in xrange(1, self.n_inputs + 1):
            self.srng.seed(seed)  # need to have same activation noise in each pass
            m, logp = self.eval_comps(x, rand=rand, const_noise=const_noise)
            idx = np.argwhere(self.input_order == i)[0, 0]
            x[:, idx] = m[:, idx] + np.exp(np.minimum(-0.5 * logp[:, idx], 10.0)) * u[:, idx]

        return x

    def calc_random_numbers(self, x):
        raise NotImplementedError()
