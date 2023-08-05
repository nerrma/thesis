#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sampler import sample_from_logreg


def l(X, y, theta):
    return -y * (X @ theta) + np.log(1 + np.exp(X @ theta))


def pi(theta):
    return np.linalg.norm(theta, 2)


def F(X, y, theta, lbd=1e-6):
    return np.sum(l(X, y, theta)) + lbd * pi(theta)


def soft_thresh(theta, lbd):
    zeros = np.where(np.abs(theta) <= lbd)[0]
    theta -= np.sign(theta) * lbd
    theta[zeros] = 0
    return theta


def nabla_F(X, y, theta, lbd=1e-6):
    return -(X.T @ y) + X.T @ ((np.exp(X @ theta)) / (1 + np.exp(X @ theta)))


def hess_F(X, y, theta, lbd=1e-6):
    expy = np.exp(X @ theta) / (1 + np.exp(X @ theta)) ** 2
    return X.T * expy @ X


def run_sim(n, p, n_iter=250):
    X, theta_star, y = sample_from_logreg(p=p, n=n)

    lbd_v = 1e-6 * n

    t = np.random.rand(p)

    theta = np.zeros(p)
    theta_p = theta
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = np.zeros((n, p))
    theta_true = np.zeros((n, p))
    theta_ns = np.zeros((n, p))
    theta_ij = np.zeros((n, p))

    y_k = np.zeros(p)
    t_k = 1

    y_k_true = np.zeros(p)
    t_k_true = 1
    theta_p_true = np.zeros(p)

    y_k_cv = np.zeros(p)
    t_k_cv = 1
    theta_p_cv = np.zeros(p)

    err_approx = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    for t in range(0, n_iter):
        f_grad = nabla_F(X, y, theta, lbd=lbd_v)
        f_hess = hess_F(X, y, theta, lbd=lbd_v)

        for i in range(0, n):
            hess_minus_i = f_hess - hess_F(
                X[i].reshape(1, -1), [y[i]], theta, lbd=lbd_v
            )

            grad_minus_i = f_grad - nabla_F(
                X[i].reshape(1, -1), [y[i]], theta, lbd=lbd_v
            )

            theta_cv[i] = soft_thresh(
                y_k_cv
                - alpha_t * grad_minus_i
                - alpha_t * hess_minus_i @ (y_k_cv - y_k),
                lbd_v * alpha_t,
            )
            t_old_cv = t_k_cv
            t_k_cv = (1 + np.sqrt(4 * t_old_cv**2 + 1)) / 2
            y_k_cv = theta_cv[i] + (t_old_cv - 1) / (t_k_cv) * (
                theta_cv[i] - theta_p_cv
            )
            theta_p_cv = theta_cv[i]

            theta_true[i] = soft_thresh(
                y_k_true
                - alpha
                * nabla_F(
                    np.delete(X, (i), axis=0),
                    np.delete(y, (i), axis=0),
                    y_k_true,
                    lbd=lbd_v,
                ),
                lbd_v * alpha_t,
            )
            t_old_true = t_k_true
            t_k_true = (1 + np.sqrt(4 * t_old_true**2 + 1)) / 2
            y_k_true = theta_true[i] + (t_old_true - 1) / (t_k_true) * (
                theta_true[i] - theta_p_true
            )
            theta_p_true = theta_true[i]

            theta_ns[i] = soft_thresh(
                y_k
                + np.linalg.inv(hess_minus_i)
                @ nabla_F(X[i].reshape(1, -1), [y[i]], theta, lbd=lbd_v),
                lbd_v * alpha_t,
            )

            theta_ij[i] = soft_thresh(
                y_k
                + np.linalg.inv(f_hess)
                @ nabla_F(X[i].reshape(1, -1), [y[i]], theta, lbd=lbd_v),
                lbd_v * alpha_t,
            )

        # actually update theta
        # theta = soft_thresh(theta - alpha * f_grad, lbd_v)
        theta = soft_thresh(y_k - alpha * f_grad, lbd_v * alpha)
        t_old = t_k
        t_k = (1 + np.sqrt(4 * t_old**2 + 1)) / 2
        y_k = theta + (t_old - 1) / (t_k) * (theta - theta_p)
        theta_p = theta

        err_approx["IACV"][t] = np.mean(
            np.linalg.norm(theta_cv - theta_true, 2, axis=1)
        )

        err_approx["NS"][t] = np.mean(np.linalg.norm(theta_ns - theta_true, 2, axis=1))
        err_approx["IJ"][t] = np.mean(np.linalg.norm(theta_ij - theta_true, 2, axis=1))
        err_approx["hat"][t] = np.mean(np.linalg.norm(theta - theta_true, 2, axis=1))

    print(np.mean(theta_cv, axis=0))
    print(np.mean(theta_true, axis=0))
    print(theta.ravel())
    print(theta_star.ravel())
    return err_approx
