import numpy as np
import matplotlib.pyplot as plt


def disp_imdata(xs, imsize, layout=(1, 1)):
    """
    Displays an array of images, a page at a time. The user can navigate pages with
    left and right arrows, start over by pressing space, or close the figure by esc.
    :param xs: an numpy array with images as rows
    :param imsize: size of the images
    :param layout: layout of images in a page
    :return: none
    """

    num_plots = np.prod(layout)
    num_xs = xs.shape[0]
    idx = [0]

    # create a figure with subplots
    fig, axs = plt.subplots(layout[0], layout[1])

    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for ax in axs:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    def plot_page():
        """Plots the next page."""

        ii = np.arange(idx[0], idx[0]+num_plots) % num_xs

        for ax, i in zip(axs, ii):
            ax.imshow(xs[i].reshape(imsize), cmap='gray', interpolation='none')
            ax.set_title(str(i))

        fig.canvas.draw()

    def on_key_event(event):
        """Event handler after key press."""

        key = event.key

        if key == 'right':
            # show next page
            idx[0] = (idx[0] + num_plots) % num_xs
            plot_page()

        elif key == 'left':
            # show previous page
            idx[0] = (idx[0] - num_plots) % num_xs
            plot_page()

        elif key == ' ':
            # show first page
            idx[0] = 0
            plot_page()

        elif key == 'escape':
            # close figure
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key_event)
    plot_page()


def probs2contours(probs, levels):
    """
    Takes an array of probabilities and produces an array of contours at specified percentile levels
    :param probs: probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    :param levels: percentile levels. have to be in [0.0, 1.0]
    :return: array of same shape as probs with percentile labels
    """

    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours


def plot_pdf_marginals(pdf, lims, gt=None, levels=(0.68, 0.95, 0.99), upper=False):
    """
    Plots marginals of a pdf, for each variable and pair of variables.
    """

    if pdf.n_dims == 1:

        fig, ax = plt.subplots(1, 1)
        xx = np.linspace(lims[0], lims[1], 200)

        pp = pdf.eval(xx[:, np.newaxis], log=False)
        ax.plot(xx, pp)
        ax.set_xlim(lims)
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        fig = plt.figure()

        lims = np.asarray(lims)
        lims = np.tile(lims, [pdf.n_dims, 1]) if lims.ndim == 1 else lims

        for i in xrange(pdf.n_dims):
            for j in xrange(i, pdf.n_dims) if upper else xrange(i + 1):

                ax = fig.add_subplot(pdf.n_dims, pdf.n_dims, i * pdf.n_dims + j + 1)

                if i == j:
                    xx = np.linspace(lims[i, 0], lims[i, 1], 500)
                    pp = pdf.eval(xx[:, np.newaxis], ii=[i], log=False)
                    ax.plot(xx, pp)
                    ax.set_xlim(lims[i])
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    if i < pdf.n_dims - 1 and not upper: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if gt is not None: ax.vlines(gt[i], 0, ax.get_ylim()[1], color='r')

                else:
                    xx = np.linspace(lims[j, 0], lims[j, 1], 200)
                    yy = np.linspace(lims[i, 0], lims[i, 1], 200)
                    X, Y = np.meshgrid(xx, yy)
                    xy = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
                    pp = pdf.eval(xy, ii=[j, i], log=False)
                    pp = pp.reshape(list(X.shape))
                    ax.contour(X, Y, probs2contours(pp, levels), levels)
                    ax.set_xlim(lims[j])
                    ax.set_ylim(lims[i])
                    if i < pdf.n_dims - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                    if j == pdf.n_dims - 1: ax.tick_params(axis='y', which='both', labelright=True)
                    if gt is not None: ax.plot(gt[j], gt[i], 'r.', ms=8)

    return fig


def plot_hist_marginals(data, weights=None, lims=None, gt=None, upper=False, rasterized=False):
    """
    Plots marginal histograms and pairwise scatter plots of a dataset.
    """

    data = np.asarray(data)
    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:

        fig, ax = plt.subplots(1, 1)
        ax.hist(data, weights=weights, bins=n_bins, normed=True, rasterized=rasterized)
        ax.set_ylim([0.0, ax.get_ylim()[1]])
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if lims is not None: ax.set_xlim(lims)
        if gt is not None: ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:

        n_dim = data.shape[1]
        fig = plt.figure()

        if weights is None:
            col = 'k'
            vmin, vmax = None, None
        else:
            col = weights
            vmin, vmax = 0., np.max(weights)

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in xrange(n_dim):
            for j in xrange(i, n_dim) if upper else xrange(i + 1):

                ax = fig.add_subplot(n_dim, n_dim, i * n_dim + j + 1)

                if i == j:
                    ax.hist(data[:, i], weights=weights, bins=n_bins, normed=True, rasterized=rasterized)
                    ax.set_ylim([0.0, ax.get_ylim()[1]])
                    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                    if i < n_dim - 1 and not upper: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if lims is not None: ax.set_xlim(lims[i])
                    if gt is not None: ax.vlines(gt[i], 0, ax.get_ylim()[1], color='r')

                else:
                    ax.scatter(data[:, j], data[:, i], c=col, s=3, marker='o', vmin=vmin, vmax=vmax, cmap='binary', edgecolors='none', rasterized=rasterized)
                    if i < n_dim - 1: ax.tick_params(axis='x', which='both', labelbottom=False)
                    if j > 0: ax.tick_params(axis='y', which='both', labelleft=False)
                    if j == n_dim - 1: ax.tick_params(axis='y', which='both', labelright=True)
                    if lims is not None:
                        ax.set_xlim(lims[j])
                        ax.set_ylim(lims[i])
                    if gt is not None: ax.scatter(gt[j], gt[i], c='r', s=12, marker='o', edgecolors='none')

    return fig


def plot_traces(xs, var_names=None):
    """
    Plots sample traces. Useful for MCMC.
    :param xs: # samples x # vars numpy array
    :param var_names: list of strings to use as names of the variables
    :return: figure handle
    """

    n_vars = xs.shape[1]
    fig, axs = plt.subplots(n_vars, 1, sharex=True)
    axs = [axs] if n_vars == 1 else axs

    for i, ax in enumerate(axs):
        ax.plot(xs[:, i])

    if var_names is not None:
        for ax, name in zip(axs, var_names):
            ax.set_ylabel(name)

    axs[-1].set_xlabel('samples')
    plt.show(block=False)

    return fig
