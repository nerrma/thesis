#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4389)


def sample_from_logreg(p=20, n=500):
    X = np.random.normal(0, 1, size=(n, p))  # sample (n, p) from standard normal
    # theta_star = np.random.normal(0, 1, size=p).reshape(-1, 1)

    theta_star = np.zeros(p).reshape(-1, 1)
    theta_star[np.random.choice(p, 5)] = np.random.normal(0, 1)

    probs = np.exp(X @ theta_star) / (np.exp(X @ theta_star) + 1)
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = np.random.binomial(1, probs[i])

    return (X, theta_star, y)


def l(X, y, theta):
    return -y * (X @ theta) + np.log(1 + np.exp(X @ theta))


def pi(theta):
    return np.linalg.norm(theta, 2)


def F(X, y, theta, lbd=1e-6):
    return np.sum(l(X, y, theta)) + lbd * pi(theta)


def nabla_F(X, y, theta, lbd=1e-6):
    return (
        -(X.T @ y) + X.T @ ((np.exp(X @ theta)) / (1 + np.exp(X @ theta))) + lbd * theta
    )


def hess_F(X, y, theta, lbd=1e-6):
    expy = np.exp(X @ theta) / (1 + np.exp(X @ theta)) ** 2
    return X.T * expy @ X + lbd * np.eye(theta.shape[0])


def run_sim(n, p, n_iter=250):
    X, theta_s, y = sample_from_logreg(p=p, n=n)

    # theta = np.random.normal(0, 1, size=(p,))
    theta = np.zeros(p)
    alpha = 0.5 / n
    alpha_t = 0.5 / n

    theta_cv = np.zeros((n, p))
    theta_true = np.zeros((n, p))
    theta_ns = np.zeros((n, p))
    theta_ij = np.zeros((n, p))

    err_approx = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    for t in range(0, n_iter):
        f_grad = nabla_F(X, y, theta)
        f_hess = hess_F(X, y, theta)
        theta = theta - alpha * f_grad
        # print(f"loss : {F(X, y, theta)}")

        for i in range(0, n):
            hess_minus_i = f_hess - hess_F(X[i].reshape(1, -1), [y[i]], theta)
            grad_minus_i = f_grad - nabla_F(X[i].reshape(1, -1), [y[i]], theta)

            # TODO, FIX IACV STEP
            theta_cv[i] = (
                theta_cv[i]
                - alpha_t * grad_minus_i
                - alpha_t * hess_minus_i @ (theta_cv[i] - theta)
            )

            theta_true[i] = theta_true[i] - alpha * nabla_F(
                np.delete(X, (i), axis=0), np.delete(y, (i), axis=0), theta_true[i]
            )

            theta_ns[i] = theta + np.linalg.inv(hess_minus_i) @ nabla_F(
                X[i].reshape(1, -1), [y[i]], theta
            )
            theta_ij[i] = theta + np.linalg.inv(f_hess) @ nabla_F(
                X[i].reshape(1, -1), [y[i]], theta
            )

        err_approx["IACV"][t] = np.mean(
            np.linalg.norm(theta_cv - theta_true, 2, axis=1)
        )
        err_approx["NS"][t] = np.mean(np.linalg.norm(theta_ns - theta_true, 2, axis=1))
        err_approx["IJ"][t] = np.mean(np.linalg.norm(theta_ij - theta_true, 2, axis=1))
        err_approx["hat"][t] = np.mean(np.linalg.norm(theta - theta_true, 2, axis=1))
    return err_approx
