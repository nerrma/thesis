#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import sklearn.datasets as datasets
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from functools import partial
import jax.numpy as jnp
from jaxopt.prox import prox_lasso
from sampler import sample_from_linear_first_5

import jax

jax.config.update("jax_platform_name", "cpu")


def l(X, y, theta):
    return -y * (X @ theta) + jnp.log(1 + jnp.exp(X @ theta))


def pi(theta):
    return jnp.linalg.norm(theta, 1)


def F_mod(theta, X, y, lbd):
    return jnp.sum(l(X, y, theta))


def run_sim(n, p, n_iter=250, lbd_v=None):
    X = pd.read_csv("data/GISETTE/gisette_train.data").to_numpy()
    y = pd.read_csv("data/GISETTE/gisette_train.labels").to_numpy()

    print(X.shape)
    print(y.shape)

    y = y.reshape(-1, 1)
    X = X / (np.linalg.norm(X, axis=0))

    n = X.shape[0]
    p = X.shape[1]
    print(f"sampled data X: {X.shape}, y: {y.shape}")

    if lbd_v is None:
        lbd_v = 1e-6 * n

    theta = np.ones((p, 1))
    theta_true = np.ones((n, p))
    theta_ij = np.ones((n, p))
    theta_ns = np.ones((n, p))
    theta_cv = np.ones((n, p))
    theta_flat = theta.flatten()
    # alpha_t = 1e-2 * (n / p) ** 2
    alpha_t = 0.25 / p
    print(f"alpha_t {alpha_t}")

    supports = []

    err_approx = {
        "IACV": np.zeros(n_iter),
        "IACV sparse": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    err_loo = {
        "IACV": np.zeros(n_iter),
        "IACV sparse": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    runtime = {
        "IACV": np.zeros(n_iter),
        "IACV sparse": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "true": np.zeros(n_iter),
    }

    nabla_F = jit(grad(F_mod))
    hess_F = jit(jacfwd(jacrev(F_mod)))

    grad_Z_f = jit(vmap(nabla_F, in_axes=(None, 0, 0, None)))
    hess_Z_f = jit(vmap(hess_F, in_axes=(None, 0, 0, None)))

    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

    mask = ~np.eye(n, dtype=bool)
    for t in range(0, n_iter):
        support = np.where(theta != 0)[0]
        supports.append(support)

        # actually update theta, coordinate descent style
        for s in support:
            X_s = X[:, s].reshape(-1, 1)
            X_theta = X @ theta
            numerator = X_s.T @ (y - X_theta + (theta[s] * X_s))
            theta[s] = prox_lasso(numerator / (X_s.T @ X_s), lbd_v / (X_s.T @ X_s))

        print(
            f"iter: {t} | loss: {F_mod(theta, X, y, lbd_v) + lbd_v * pi(theta)} | support size: {len(support)}"
        )

        theta_flat = theta.flatten()
        X_s = np.zeros(X.shape)
        flat_support = np.where(theta_flat != 0)[0]
        X_s[:, flat_support] = X[:, flat_support]

        # IACV block
        iacv_start = time.time()
        f_grad = jnp.nan_to_num(nabla_F(theta_flat, X_s, y, lbd_v), nan=lbd_v)
        f_hess = jnp.nan_to_num(hess_F(theta_flat, X_s, y, lbd_v), nan=lbd_v)

        grad_Z = jnp.nan_to_num(grad_Z_f(theta_flat, X_s, y, lbd_v), nan=lbd_v)
        hess_Z = jnp.nan_to_num(hess_Z_f(theta_flat, X_s, y, lbd_v), nan=lbd_v)

        grad_minus = f_grad - grad_Z
        hess_minus = f_hess - hess_Z

        # print(theta_cv[0])
        theta_cv = prox_lasso(
            theta_cv
            - alpha_t * grad_minus
            - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta_flat)),
            lbd_v * (n / p),
        )

        print(
            f"IACV grad norm {np.linalg.norm(grad_minus)} | hess norm {np.linalg.norm(hess_minus)}"
        )
        iacv_end = time.time()

        # NS block
        ns_start = time.time()
        theta_flat = theta.flatten()
        f_hess_s = jnp.nan_to_num(hess_F(theta_flat, X_s, y, lbd_v), nan=lbd_v)

        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta, X_s, y, lbd_v), nan=lbd_v)
        hess_Z_s = jnp.nan_to_num(hess_Z_f(theta_flat, X_s, y, lbd_v), nan=lbd_v)

        hess_minus_s = f_hess_s - hess_Z_s
        theta_ns = theta + jnp.nan_to_num(
            vmap_matmul(jnp.linalg.inv(hess_minus_s), grad_Z_s), nan=lbd_v
        )
        ns_end = time.time()

        # IJ block
        ij_start = time.time()

        theta_flat = theta.flatten()
        f_hess_s = jnp.nan_to_num(hess_F(theta_flat, X_s, y, lbd_v), nan=lbd_v)
        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta, X_s, y, lbd_v), nan=lbd_v)

        f_hess_s = (X @ f_hess_s).T @ X

        # print(f"grad norm {np.linalg.norm(grad_Z_s)} | hess norm {np.linalg.norm(f_hess_s)}")

        theta_ij = theta + jnp.nan_to_num(
            jit(vmap(jnp.matmul, in_axes=(None, 0)))(
                jnp.linalg.inv(f_hess_s), grad_Z_s
            ),
            nan=lbd_v,
        )
        theta_ij = prox_lasso(theta_ij, lbd_v)
        ij_end = time.time()

        # true block
        true_start = time.time()
        for i in range(n):
            X_tmp = X[mask[i, :]]
            y_tmp = y[mask[i, :]]
            tt = theta_true[i].reshape(-1, 1)
            sub_support = np.where(theta_true[i] != 0)[0]
            for s in sub_support:
                X_s = X_tmp[:, s].reshape(-1, 1)
                X_theta = X_tmp @ tt
                numerator = X_s.T @ (y_tmp - X_theta + (tt[s] * X_s))
                theta_true[i][s] = prox_lasso(
                    numerator / (X_s.T @ X_s), lbd_v / (X_s.T @ X_s)
                )

        true_end = time.time()

        theta_ij = theta_ij.reshape(n, p)
        theta_ns = theta_ns.reshape(n, p)
        true_stack = jnp.stack(theta_true)
        for k, v in list(
            [
                ("NS", theta_ns),
                ("IJ", theta_ij),
                ("hat", theta_flat),
                ("IACV", theta_cv),
            ],
        ):
            err_approx[k][t] = jnp.mean(jnp.linalg.norm(v - true_stack, 2, axis=1))

        true_LOO = F_mod(true_stack.T, X, y, lbd_v)
        for k, v in list(
            [
                ("NS", theta_ns),
                ("IJ", theta_ij),
                ("hat", theta_flat),
                ("IACV", theta_cv),
            ],
        ):
            err_loo[k][t] = (
                jnp.mean(jnp.abs(F_mod(v.T, X, y, lbd_v) - true_LOO)) / true_LOO
            )

        ## track runtimes
        for k, (t1, t2) in list(
            [
                ("NS", (ns_start, ns_end)),
                ("IJ", (ij_start, ij_end)),
                ("IACV", (iacv_start, iacv_end)),
                ("true", (true_start, true_end)),
            ]
        ):
            runtime[k][t] = runtime[k][t - 1] + t2 - t1 if t > 0 else 0

    params = {
        # "IACV": theta_cv,
        # "IACV sparse": theta_cv_sparse,
        # "IJ": theta_ij,
        # "NS": theta_ns,
        # "true": theta_true,
        "star": theta_star,
    }

    for i in range(5):
        print(f"when excluding {i + 1}")
        print(f"flat {theta_flat[:5]} | support {len(np.where(theta_flat != 0)[0])}")
        print(
            f"true {theta_true[i][:5]} | support {len(np.where(theta_true[i] != 0)[0])}"
        )
        print(f"IJ   {theta_ij[i][:5]} | support {len(np.where(theta_ij[i] != 0)[0])}")
        print(f"NS   {theta_ns[i][:5]} | support {len(np.where(theta_ns[i] != 0)[0])}")
        print(f"IACV {theta_cv[i][:5]} | support {len(np.where(theta_cv[i] != 0)[0])}")
    return err_approx, err_loo, runtime, supports, params


if "__main__":
    n = 50
    p = 150
    # n = 50
    # p = 200
    # special_lambda = 1.5 * np.sqrt(np.log(p) / n)
    special_lambda = n / p * np.sqrt(np.log(p) / n)
    print(f"special_lambda is {special_lambda}")
    err_approx, err_loo, runtime, supports, params = run_sim(
        n=n, p=p, n_iter=100, lbd_v=special_lambda
    )

    print(f"hat final error: {err_approx['hat'][-1]}")
    print(f"IJ final error: {err_approx['IJ'][-1]} | runtime {runtime['IJ'][-1]}")
    print(f"NS final error: {err_approx['NS'][-1]} | runtime {runtime['NS'][-1]}")
    print(f"IACV final error: {err_approx['IACV'][-1]} | runtime {runtime['IACV'][-1]}")

    print(f"hat final LOO: {err_loo['hat'][-1]}")
    print(f"IJ final LOO: {err_loo['IJ'][-1]}")
    print(f"NS final LOO: {err_loo['NS'][-1]}")
    print(f"IACV final LOO: {err_loo['IACV'][-1]}")
    print(f"true CV runtime {runtime['true'][-1]}")
