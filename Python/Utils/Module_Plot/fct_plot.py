# -*- coding: utf-8 -*-


import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.optimize import least_squares
from scipy import linalg
import seaborn as sns

## TODO comment PaperPlot class


def powercutoff_fit(x, a, b, c, d):
    return a * x ** b * np.exp(-(x / c) ** d)


def power_fit(x, a, b):
    return a * x ** b


def lin_fit(x, a, b):
    return a + x * b


def lognorm_fit(x, a, b):
    return 1 / (x * b * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - a) ** 2 / (2 * b ** 2))


def gauss_fit(x, a, b):
    return 1 / (b * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - a) / b) ** 2)


def fit_powerlaw(x, n, n_cutoff=1e-30, start_x=1e2, plot=True, f_scale=1.0, function_to_fit='power_cut_off'):
    """
    fit n(x) = a*x**b*exp(-(x/c)**d) for n > n_cutoff (which defines a maximum x to fit) and x > start_x.
    Loss function is soft_l1. Adjust f_scale to weight more or less smaller values.
    Adjust x0 directly in code.
    start_x : cut off little value of x => noise
    n_cutoff : cut large value of x if n to small
    """
    mask = n > n_cutoff
    stop_x = x[np.where(n > n_cutoff)[0][-1]]
    xs = x[mask]
    n_f = n[mask]

    start = np.where(xs > start_x)[0][0]
    stop = np.where(xs < stop_x)[0][-1]

    print(f"start x = {start_x:.2e}")
    print(f"stop x = {stop_x:.2e}")

    if stop - start < 2:
        raise ValueError

    def residuals(params):
        if function_to_fit == 'power_cut_off':
            return np.log(powercutoff_fit(xs[start:stop], *params)) - np.log(n_f[start:stop])
        else:
            return np.log(power_fit(xs[start:stop], *params)) - np.log(n_f[start:stop])

    if function_to_fit == 'power_cut_off':
        result = least_squares(residuals, x0=(n_f[0], -1, stop_x / 2, 1), loss="soft_l1", max_nfev=10000,
                               bounds=[[0, -np.inf, 0, 0], [np.inf, 0, stop_x * 1e4, np.inf]], f_scale=f_scale)
        names = ["a", "b", "c", "d"]
        print(result.message)
        print("n(x) = a*x**b*exp(-(x/c)**d)")
        for i in range(len(result.x)):
            print(f"{names[i]} = {result.x[i]:.2e}")

        U, s, Vh = linalg.svd(result.jac, full_matrices=False)
        tol = np.finfo(float).eps * s[0] * max(result.jac.shape)
        w = s > tol
        cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
        perr = np.sqrt(np.diag(cov))
        print("perr = ", perr)

    else:
        result = least_squares(residuals, x0=(n_f[0], -1), loss="soft_l1", max_nfev=10000,
                               bounds=[[0, -np.inf], [np.inf, 0]], f_scale=f_scale)
        names = ["a", "b"]
        print(result.message)
        print("n(x) = a*x**b")
        for i in range(len(result.x)):
            print(f"{names[i]} = {result.x[i]:.2e}")
        U, s, Vh = linalg.svd(result.jac, full_matrices=False)
        tol = np.finfo(float).eps * s[0] * max(result.jac.shape)
        w = s > tol
        cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
        perr = np.sqrt(np.diag(cov))
        print("perr = ", perr)

    return result, xs[start::], n_f, perr


def fit_lin(x, n, start_x=1e2, n_cutoff=1e-30, f_scale=1.0, function_to_fit='power_law'):
    """
    fit n(x) = a*x**b*exp(-(x/c)**d) for n > n_cutoff (which defines a maximum x to fit) and x > start_x.
    Loss function is soft_l1. Adjust f_scale to weight more or less smaller values.
    Adjust x0 directly in code.
    start_x : cut off little value of x => noise
    n_cutoff : cut large value of x if n to small
    """
    ## cutoff mask
    x = x[np.where(n > n_cutoff)[0]]
    n = n[np.where(n > n_cutoff)[0]]

    n_f = n[x > start_x]
    xs = x[x > start_x]

    print(f"start x = {start_x:.2e}")
    print(f"start y = {n_cutoff:.2e}")

    def residuals(params):
        if function_to_fit == 'power_law':
            return np.log(power_fit(xs, *params)) - np.log(n_f)
        else:
            return lin_fit(xs, *params) - n_f

    result = least_squares(residuals, x0=(np.min(n_f), 1), loss="soft_l1", max_nfev=10000,
                           bounds=[[0, 0], [np.inf, np.inf]], f_scale=f_scale)
    names = ["a", "b"]
    print(result.message)
    print("n(x) = a*x**b")
    for i in range(len(result.x)):
        print(f"{names[i]} = {result.x[i]:.2e}")

    U, s, Vh = linalg.svd(result.jac, full_matrices=False)
    tol = np.finfo(float).eps * s[0] * max(result.jac.shape)
    w = s > tol
    cov = (Vh[w].T / s[w] ** 2) @ Vh[w]  # robust covariance matrix
    perr = np.sqrt(np.diag(cov))
    print("perr = ", perr)

    return result, xs, n_f, perr


def plot_shift_f(f, var=False):
    if not var:
        return f - np.mean(f)
    else:
        return (f - np.mean(f)) / np.sqrt(np.var(f))

def zero_shift_f(f, idx=0, var=False):
    if not var:
        return f - f[idx]
    else:
        return (f - f[idx]) / np.sqrt(np.var(f))

def plot_scale_f(f):
    return f / np.sqrt(np.var(f))
