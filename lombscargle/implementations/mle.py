"""Tools for maximum likelihood estimation associated with Lomb-Scargle"""
import numpy as np


def design_matrix(t, frequency, dy=None, bias=True, nterms=1):
    """Compute the Lomb-Scargle design matrix at the given frequency

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    t = np.asarray(t)
    assert t.ndim == 1
    assert np.isscalar(frequency)

    if bias:
        cols = [np.ones_like(t)]
    else:
        cols = []

    for i in range(1, nterms + 1):
        cols.append(np.sin(2 * np.pi * i * frequency * t))
        cols.append(np.cos(2 * np.pi * i * frequency * t))
    XT = np.vstack(cols)

    if dy is not None:
        XT /= dy

    return np.transpose(XT)


def periodic_fit(self, t, y, dy, frequency, t_fit,
                 center_data=True, fit_bias=True):
    """Compute the Lomb-Scargle model fit at a given frequency

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    if dy is None:
        dy = 1

    t, y, dy = np.broadcast_arrays(t, y, dy)
    t_fit = np.asarray(t_fit)
    assert t.ndim == 1
    assert t_fit.ndim == 1
    assert np.isscalar(frequency)

    w = dy ** -2.0
    w /= w.sum()

    if center_data:
        yw = (y - np.dot(w, y)) / dy
    else:
        yw = y / dy
    chi2_ref = np.dot(yw, yw)

    X = design_matrix(t, frequency, dy=dy, bias=fit_bias)
    X_fit = design_matrix(t_fit, frequency, bias=fit_bias)

    theta_MLE = np.linalg.solve(np.dot(X.T, X),
                                np.dot(X.T, yw))

    return np.dot(X_fit, theta_MLE)
