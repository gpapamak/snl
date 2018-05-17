from itertools import izip
import numpy as np
import scipy.misc
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import util.math

from pdfs.gaussian import Gaussian


class MoG:
    """
    Implements a mixture of gaussians.
    """

    def __init__(self, a, ms=None, Ps=None, Us=None, Ss=None, xs=None):
        """
        Creates a mog with a valid combination of parameters or an already given list of gaussian variables.
        :param a: mixing coefficients
        :param ms: means
        :param Ps: precisions
        :param Us: precision factors such that U'U = P
        :param Ss: covariances
        :param xs: list of gaussian variables
        """

        if ms is not None:

            if Ps is not None:
                self.xs = [Gaussian(m=m, P=P) for m, P in izip(ms, Ps)]

            elif Us is not None:
                self.xs = [Gaussian(m=m, U=U) for m, U in izip(ms, Us)]

            elif Ss is not None:
                self.xs = [Gaussian(m=m, S=S) for m, S in izip(ms, Ss)]

            else:
                raise ValueError('Precision information missing.')

        elif xs is not None:
            self.xs = xs

        else:
            raise ValueError('Mean information missing.')

        self.a = np.asarray(a)
        self.n_dims = self.xs[0].n_dims
        self.n_components = len(self.xs)

    def gen(self, n_samples=None, return_comps=False, rng=np.random):
        """
        Generates independent samples from mog.
        """

        if n_samples is None:

            i = util.math.discrete_sample(self.a, rng=rng)
            sample = self.xs[i].gen(rng=rng)

            return (sample, i) if return_comps else sample

        else:

            samples = np.empty([n_samples, self.n_dims])
            ii = util.math.discrete_sample(self.a, n_samples, rng)
            for i, x in enumerate(self.xs):
                idx = ii == i
                N = np.sum(idx.astype(int))
                samples[idx] = x.gen(N, rng=rng)

            return (samples, ii) if return_comps else samples

    def eval(self, x, ii=None, log=True):
        """
        Evaluates the mog pdf.
        :param x: rows are inputs to evaluate at
        :param ii: a list of indices specifying which marginal to evaluate. if None, the joint pdf is evaluated
        :param log: if True, the log pdf is evaluated
        :return: pdf or log pdf
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.eval(x[np.newaxis, :], ii, log)[0]

        ps = np.array([c.eval(x, ii, log) for c in self.xs]).T
        res = scipy.misc.logsumexp(ps + np.log(self.a), axis=1) if log else np.dot(ps, self.a)

        return res

    def grad_log_p(self, x):
        """
        Evaluates the gradient of the log mog pdf.
        :param x: rows are inputs to evaluate at
        :return: d/dx log p(x)
        """

        x = np.asarray(x)

        if x.ndim == 1:
            return self.grad_log_p(x[np.newaxis, :])[0]

        ps = np.array([c.eval(x, log=True) for c in self.xs])
        ws = util.math.softmax(ps.T + np.log(self.a)).T
        ds = np.array([c.grad_log_p(x) for c in self.xs])

        res = np.sum(ws[:, :, np.newaxis] * ds, axis=0)

        return res

    def __mul__(self, other):
        """
        Multiply with a single gaussian.
        """

        assert isinstance(other, Gaussian)

        ys = [x * other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(izip(self.xs, ys)):

            lcs[i] = x.logdetP + other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) + np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __imul__(self, other):
        """
        Incrementally multiply with a single gaussian.
        """

        assert isinstance(other, Gaussian)

        res = self * other

        self.a = res.a
        self.xs = res.xs

        return res

    def __div__(self, other):
        """
        Divide by a single gaussian.
        """

        assert isinstance(other, Gaussian)

        ys = [x / other for x in self.xs]

        lcs = np.empty_like(self.a)

        for i, (x, y) in enumerate(izip(self.xs, ys)):

            lcs[i] = x.logdetP - other.logdetP - y.logdetP
            lcs[i] -= np.dot(x.m, np.dot(x.P, x.m)) - np.dot(other.m, np.dot(other.P, other.m)) - np.dot(y.m, np.dot(y.P, y.m))
            lcs[i] *= 0.5

        la = np.log(self.a) + lcs
        la -= scipy.misc.logsumexp(la)
        a = np.exp(la)

        return MoG(a=a, xs=ys)

    def __idiv__(self, other):
        """
        Incrementally divide by a single gaussian.
        """

        assert isinstance(other, Gaussian)

        res = self / other

        self.a = res.a
        self.xs = res.xs

        return res

    def calc_mean_and_cov(self):
        """
        Calculate the mean vector and the covariance matrix of the mog.
        """

        ms = [x.m for x in self.xs]
        m = np.dot(self.a, np.array(ms))

        msqs = [x.S + np.outer(mi, mi) for x, mi in izip(self.xs, ms)]
        S = np.sum(np.array([a * msq for a, msq in izip(self.a, msqs)]), axis=0) - np.outer(m, m)

        return m, S

    def project_to_gaussian(self):
        """
        Returns a gaussian with the same mean and precision as the mog.
        """

        m, S = self.calc_mean_and_cov()
        return Gaussian(m=m, S=S)

    def prune_negligible_components(self, threshold):
        """
        Removes all the components whose mixing coefficient is less than a threshold.
        """

        ii = np.nonzero((self.a < threshold).astype(int))[0]
        total_del_a = np.sum(self.a[ii])
        del_count = ii.size

        self.n_components -= del_count
        self.a = np.delete(self.a, ii)
        self.a += total_del_a / self.n_components
        self.xs = [x for i, x in enumerate(self.xs) if i not in ii]

    def kl(self, other, n_samples=10000, rng=np.random):
        """
        Estimates the kl from this to another pdf, i.e. KL(this | other), using monte carlo.
        """

        x = self.gen(n_samples, rng=rng)
        lp = self.eval(x, log=True)
        lq = other.eval(x, log=True)
        t = lp - lq

        res = np.mean(t)
        err = np.std(t, ddof=1) / np.sqrt(n_samples)

        return res, err


def fit_mog(x, n_components, w=None, tol=1.0e-9, maxiter=float('inf'), verbose=False, rng=np.random):
    """
    Fit and return a mixture of gaussians to (possibly weighted) data using expectation maximization.
    """

    x = x[:, np.newaxis] if x.ndim == 1 else x
    n_data, n_dim = x.shape

    # initialize
    a = np.ones(n_components) / n_components
    ms = rng.randn(n_components, n_dim)
    Ss = [np.eye(n_dim) for _ in xrange(n_components)]
    iter = 0

    # calculate log p(x,z), log p(x) and total log likelihood
    logPxz = np.array([scipy.stats.multivariate_normal.logpdf(x, ms[k], Ss[k]) for k in xrange(n_components)])
    logPxz += np.log(a)[:, np.newaxis]
    logPx = scipy.misc.logsumexp(logPxz, axis=0)
    loglik_prev = np.mean(logPx) if w is None else np.dot(w, logPx)

    while True:

        # e step
        z = np.exp(logPxz - logPx)

        # m step
        if w is None:
            Nk = np.sum(z, axis=1)
            a = Nk / n_data
            ms = np.dot(z, x) / Nk[:, np.newaxis]
            for k in xrange(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * z[k], xm) / Nk[k]
        else:
            zw = z * w
            a = np.sum(zw, axis=1)
            ms = np.dot(zw, x) / a[:, np.newaxis]
            for k in xrange(n_components):
                xm = x - ms[k]
                Ss[k] = np.dot(xm.T * zw[k], xm) / a[k]

        # calculate log p(x,z), log p(x) and total log likelihood
        logPxz = np.array([scipy.stats.multivariate_normal.logpdf(x, ms[k], Ss[k]) for k in xrange(n_components)])
        logPxz += np.log(a)[:, np.newaxis]
        logPx = scipy.misc.logsumexp(logPxz, axis=0)
        loglik = np.mean(logPx) if w is None else np.dot(w, logPx)

        # check progress
        iter += 1
        diff = loglik - loglik_prev
        assert diff >= 0.0, 'Log likelihood decreased! There is a bug somewhere!'
        if verbose: print 'Iteration = {0}, log likelihood = {1}, diff = {2}'.format(iter, loglik, diff)
        if diff < tol or iter > maxiter: break
        loglik_prev = loglik

    return MoG(a=a, ms=ms, Ss=Ss)


def make_2d_cov(stds, theta):
    """
    Creates a 2d covariance matrix given a list of standard deviations and an angle.
    """

    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    C = R * np.array(stds)
    return np.dot(C, C.T)


def prec2ellipse(P, Pm):
    """
    Given a precision matrix and a precision-mean product, it returns the parameters of a 2d ellipse.
    """

    cov = np.linalg.inv(P)
    m = np.dot(cov, Pm)
    s2, W = np.linalg.eig(cov)
    s = np.sqrt(s2)
    theta = np.arctan(W[1, 0] / W[0, 0])

    return m, s[0], s[1], theta


def test_mog():
    """
    Constructs a random mog, samples from it and projects it to a gaussian. Then it's modified by multiplication with
    one of its components and done the same. The results are visualized as a test for correctness.
    """

    rng = np.random

    # parameters of a 2d mog
    n_components = rng.randint(1, 6)
    a = np.exp(0.5 * rng.randn(n_components))
    a /= np.sum(a)
    ms = np.zeros([n_components, 2])
    Ss = np.zeros([n_components, 2, 2])
    for i in xrange(n_components):
        ms[i] = 4.0 * rng.randn(2)
        Ss[i] = make_2d_cov(np.exp(rng.rand(2)), 2.0 * np.pi * rng.rand())

    # construct the mog, project it and sample from it
    mog = MoG(a=a, ms=ms, Ss=Ss)
    gaussian = mog.project_to_gaussian()
    samples = mog.gen(10000)
    xlim = [np.min(samples[:, 0]) - 1.0, np.max(samples[:, 0]) + 1.0]
    ylim = [np.min(samples[:, 1]) - 1.0, np.max(samples[:, 1]) + 1.0]
    xx = np.linspace(*xlim, num=200)
    yy = np.linspace(*ylim, num=200)

    # plot mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0, 1, mog.n_components)]
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)

    # plot projected gaussian and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    m, s1, s2, theta = prec2ellipse(gaussian.P, gaussian.Pm)
    eli = Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, lw=6)
    ax.add_artist(eli)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, gaussian.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, gaussian.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)

    # now modify the mog, project it and sample from it
    mog *= mog.xs[0]
    gaussian = mog.project_to_gaussian()
    samples = mog.gen(10000)

    # plot mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)

    # plot projected gaussian and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    m, s1, s2, theta = prec2ellipse(gaussian.P, gaussian.Pm)
    eli = Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, lw=6)
    ax.add_artist(eli)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, gaussian.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, gaussian.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)

    plt.show()


def test_em_mog():
    """
    Test the em algorithm for mog.
    """

    rng = np.random

    # parameters of a 2d mog
    n_components = rng.randint(1, 6)
    a = np.exp(0.5 * rng.randn(n_components))
    a /= np.sum(a)
    ms = np.zeros([n_components, 2])
    Ss = np.zeros([n_components, 2, 2])
    for i in xrange(n_components):
        ms[i] = 4.0 * rng.randn(2)
        Ss[i] = make_2d_cov(np.exp(rng.rand(2)), 2.0 * np.pi * rng.rand())

    # construct the mog and sample from it
    mog = MoG(a=a, ms=ms, Ss=Ss)
    samples = mog.gen(5000)
    xlim = [np.min(samples[:, 0]) - 1.0, np.max(samples[:, 0]) + 1.0]
    ylim = [np.min(samples[:, 1]) - 1.0, np.max(samples[:, 1]) + 1.0]
    xx = np.linspace(*xlim, num=200)
    yy = np.linspace(*ylim, num=200)

    # plot mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0, 1, mog.n_components)]
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)
    fig.suptitle('original')

    # fit another mog to the samples
    mog_em = fit_mog(samples, n_components=n_components, verbose=True)
    kl, err = mog.kl(mog_em)
    print 'KL(mog | fitted mog) = {0} +/- {1}'.format(kl, 3.0 * err)

    # plot fitted mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0, 1, mog_em.n_components)]
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog_em.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog_em.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog_em.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog_em.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)
    fig.suptitle('fitted')

    plt.show()


def test_weighted_em_mog():
    """
    Test the em algorithm for mog with a weighted dataset.
    """

    rng = np.random

    # parameters of a 2d mog
    n_components = rng.randint(1, 6)
    a = np.exp(0.5 * rng.randn(n_components))
    a /= np.sum(a)
    ms = np.zeros([n_components, 2])
    Ss = np.zeros([n_components, 2, 2])
    for i in xrange(n_components):
        ms[i] = 4.0 * rng.randn(2)
        Ss[i] = make_2d_cov(np.exp(rng.rand(2)), 2.0 * np.pi * rng.rand())

    # construct the mog, project it to a gaussian, sample from the gaussian and weight the samples
    mog = MoG(a=a, ms=ms, Ss=Ss)
    prop = mog.project_to_gaussian()
    samples = prop.gen(5000)
    logweights = mog.eval(samples, log=True) - prop.eval(samples, log=True)
    logweights -= scipy.misc.logsumexp(logweights)
    xlim = [np.min(samples[:, 0]) - 1.0, np.max(samples[:, 0]) + 1.0]
    ylim = [np.min(samples[:, 1]) - 1.0, np.max(samples[:, 1]) + 1.0]
    xx = np.linspace(*xlim, num=200)
    yy = np.linspace(*ylim, num=200)

    # plot mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0, 1, mog.n_components)]
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)
    fig.suptitle('original')

    # fit another mog to the weighted samples
    mog_em = fit_mog(samples, w=np.exp(logweights), n_components=n_components, verbose=True)
    kl, err = mog.kl(mog_em)
    print 'KL(mog | fitted mog) = {0} +/- {1}'.format(kl, 3.0 * err)

    # plot fitted mog and samples
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.plot(samples[:, 0], samples[:, 1], '.', ms=1)
    cmap = plt.get_cmap('rainbow')
    cols = [cmap(i) for i in np.linspace(0, 1, mog_em.n_components)]
    eli_params = [prec2ellipse(x.P, x.Pm) for x in mog_em.xs]
    elis = [Ellipse(xy=m, width=2*s1, height=2*s2, angle=theta/np.pi*180.0, fill=False, ec=col, lw=6) for (m, s1, s2, theta), col in izip(eli_params, cols)]
    [ax.add_artist(eli) for eli in elis]
    ax.legend(elis, map(str, mog_em.a))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax = fig.add_subplot(132)
    ax.plot(xx, mog_em.eval(xx[:, np.newaxis], ii=[0], log=False))
    ax.set_xlim(xlim)
    ax = fig.add_subplot(133)
    ax.plot(yy, mog_em.eval(yy[:, np.newaxis], ii=[1], log=False))
    ax.set_xlim(ylim)
    fig.suptitle('fitted')

    plt.show()
