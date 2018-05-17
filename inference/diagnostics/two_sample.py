import itertools
import numpy as np


def two_sample_test_classifier(x0, x1, rng=np.random):
    """
    Classifier-based two sample test. Given two datasets, trains a binary classifier to discriminate between them, and
    reports how well it does.
    :param x0: first dataset
    :param x1: second dataset
    :param rng: random generator to use
    :return: discrimination accuracy
    """

    import ml.models.neural_nets as nn
    import ml.trainers as trainers
    import ml.loss_functions as lf

    # create dataset
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    n_x0, n_dims = x0.shape
    n_x1 = x1.shape[0]
    n_data = n_x0 + n_x1
    assert n_dims == x1.shape[1], 'inconsistent sizes'
    xs = np.vstack([x0, x1])
    ys = np.hstack([np.zeros(n_x0), np.ones(n_x1)])

    # split in training / validation sets
    n_val = int(n_data * 0.1)
    xs_val, ys_val = xs[:n_val], ys[:n_val]
    xs_trn, ys_trn = xs[n_val:], ys[n_val:]

    # create classifier
    classifier = nn.FeedforwardNet(n_dims)
    classifier.addLayer(n_dims * 10, 'relu', rng=rng)
    classifier.addLayer(n_dims * 10, 'relu', rng=rng)
    classifier.addLayer(1, 'logistic', rng=rng)

    # train classifier
    trn_target, trn_loss = lf.CrossEntropy(classifier.output)
    val_target, val_loss = lf.CrossEntropy(classifier.output)
    trainer = trainers.SGD(
        model=classifier,
        trn_data=[xs_trn, ys_trn],
        trn_loss=trn_loss,
        trn_target=trn_target,
        val_data=[xs_val, ys_val],
        val_loss=val_loss,
        val_target=val_target
    )
    trainer.train(
        minibatch=100,
        patience=20,
        monitor_every=1,
        logger=None
    )

    # measure accuracy
    pred = classifier.eval(xs)[:, 0] > 0.5
    acc = np.mean(pred == ys)

    return acc


def sq_maximum_mean_discrepancy(xs, ys, wxs=None, wys=None, scale=None, return_scale=False):
    """
    Finite sample estimate of square maximum mean discrepancy. Uses a gaussian kernel.
    :param xs: first sample
    :param ys: second sample
    :param wxs: weights for first sample, optional
    :param wys: weights for second sample, optional
    :param scale: kernel scale. If None, calculate it from data
    :return: squared mmd, scale if not given
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n_x = xs.shape[0]
    n_y = ys.shape[0]

    if wxs is not None:
        wxs = np.asarray(wxs)
        assert wxs.ndim == 1 and n_x == wxs.size

    if wys is not None:
        wys = np.asarray(wys)
        assert wys.ndim == 1 and n_y == wys.size

    xx_sq_dists = np.sum(np.array([x1 - x2 for x1, x2 in itertools.combinations(xs, 2)]) ** 2, axis=1)
    yy_sq_dists = np.sum(np.array([y1 - y2 for y1, y2 in itertools.combinations(ys, 2)]) ** 2, axis=1)
    xy_sq_dists = np.sum(np.array([x1 - y2 for x1, y2 in itertools.product(xs, ys)]) ** 2, axis=1)

    scale = np.median(np.sqrt(np.concatenate([xx_sq_dists, yy_sq_dists, xy_sq_dists]))) if scale is None else scale
    c = -0.5 / (scale ** 2)

    if wxs is None:
        kxx = np.sum(np.exp(c * xx_sq_dists)) / (n_x * (n_x - 1))
    else:
        wxx = np.array([w1 * w2 for w1, w2 in itertools.combinations(wxs, 2)])
        kxx = np.sum(wxx * np.exp(c * xx_sq_dists)) / (1.0 - np.sum(wxs ** 2))

    if wys is None:
        kyy = np.sum(np.exp(c * yy_sq_dists)) / (n_y * (n_y - 1))
    else:
        wyy = np.array([w1 * w2 for w1, w2 in itertools.combinations(wys, 2)])
        kyy = np.sum(wyy * np.exp(c * yy_sq_dists)) / (1.0 - np.sum(wys ** 2))

    if wxs is None and wys is None:
        kxy = np.sum(np.exp(c * xy_sq_dists)) / (n_x * n_y)
    else:
        if wxs is None:
            wxs = np.full(n_x, 1.0 / n_x)
        if wys is None:
            wys = np.full(n_y, 1.0 / n_y)
        wxy = np.array([w1 * w2 for w1, w2 in itertools.product(wxs, wys)])
        kxy = np.sum(wxy * np.exp(c * xy_sq_dists))

    mmd2 = 2 * (kxx + kyy - kxy)

    if return_scale:
        return mmd2, scale
    else:
        return mmd2


def two_sample_test_kernel(xs, ys, rng=np.random):
    """
    Kernel-based two-sample test. Calculates squared maximum mean discrepancy between datasets.
    :param xs: first sample
    :param ys: second sample
    :param rng: random number generator to use
    :return: squared mmd, standard error
    """

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n_x = xs.shape[0]
    n_y = ys.shape[0]

    mmd2, scale = sq_maximum_mean_discrepancy(xs, ys, return_scale=True)

    n_bs_samples = 20
    mmd2_bs = np.empty(n_bs_samples + 1)
    mmd2_bs[-1] = mmd2

    for i in xrange(n_bs_samples):
        ix = rng.randint(0, n_x, n_x)
        iy = rng.randint(0, n_y, n_y)
        mmd2_bs[i] = sq_maximum_mean_discrepancy(xs[ix], ys[iy], scale=scale)

    std_err = np.std(mmd2_bs)

    return mmd2, std_err


def test_two_sample_test_classifier(rng=np.random):
    """
    A test for the classifier-based two sample test.
    """

    x0 = rng.randn(1000, 5)
    x1 = rng.randn(1000, 5)
    res = two_sample_test_classifier(x0, x1)
    print 'Same distribution: {0:.2%}'.format(res)

    x0 = rng.randn(1000, 5) - 0.5
    x1 = rng.randn(1000, 5) + 0.5
    res = two_sample_test_classifier(x0, x1)
    print 'Different distributions: {0:.2%}'.format(res)


def test_two_sample_test_kernel(rng=np.random):
    """
    A test for the kernel-based two sample test.
    """

    xs = rng.randn(1000, 5)
    ys = rng.randn(1000, 5)
    res, err = two_sample_test_kernel(xs, ys)
    print 'Same distribution: {0:.2} +/- {1:.2}'.format(res, 2 * err)

    xs = rng.randn(1000, 5) - 0.5
    ys = rng.randn(1000, 5) + 0.5
    res, err = two_sample_test_kernel(xs, ys)
    print 'Different distributions: {0:.2} +/- {1:.2}'.format(res, 2 * err)
