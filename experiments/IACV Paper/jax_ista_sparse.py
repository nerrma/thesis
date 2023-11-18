#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from functools import partial
import jax.numpy as jnp
from jaxopt.prox import prox_lasso
from sampler import sample_from_logreg

import jax

jax.config.update("jax_platform_name", "cpu")


def l(X, y, theta):
    return -y * (X @ theta) + jnp.log(1 + jnp.exp(X @ theta))


def l_s(X, y, theta):
    return -y * jnp.sum(X * theta) + jnp.log(1 + jnp.exp(jnp.sum(X * theta)))


def pi(theta):
    return jnp.linalg.norm(theta, 1)


# def F_mod(theta, X, y, lbd):
#    return jnp.sum(l(X, y, theta)) + lbd * pi(theta)


def F_mod(theta, X, y, lbd):
    return jnp.sum(l(X, y, theta))


def F_mod_s(theta, X, y, lbd):
    return jnp.sum(l_s(X, y, theta)) + lbd * pi(theta)


def run_sim(n, p, n_iter=250, lbd_v=None):
    X, _, y = sample_from_logreg(p=p, n=n)

    if lbd_v is None:
        lbd_v = 1e-6 * n

    theta = jnp.ones(p)
    alpha = 0.5 / n
    alpha_t = alpha

    # theta_cv = jnp.ones((n, p))
    theta_cv = np.ones((n, p))
    theta_cv_sparse = jnp.ones((n, p))
    theta_true = [jnp.ones(p)] * n
    # theta_ns = jnp.ones((n, p))
    theta_ns = np.zeros((n, p))
    # theta_ij = jnp.ones((n, p))
    theta_ij = np.zeros((n, p))

    supports = []

    err_approx = {
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

    nabla_F_s = jit(grad(F_mod_s))
    hess_F_s = jit(jacfwd(jacrev(F_mod_s)))

    grad_Z_f_s = jit(vmap(nabla_F_s, in_axes=(None, 0, 0, None)))
    hess_Z_f_s = jit(vmap(hess_F_s, in_axes=(None, 0, 0, None)))

    mask = ~np.eye(n, dtype=bool)
    for t in range(0, n_iter):
        support = np.where(theta != 0)[0]
        supports.append(support)

        c_support = np.where(theta == 0)[0]

        # IACV block
        iacv_start = time.time()
        f_grad = jnp.nan_to_num(nabla_F(theta, X, y, lbd_v), nan=lbd_v)
        f_hess = jnp.nan_to_num(hess_F(theta, X, y, lbd_v), nan=lbd_v)

        grad_Z = jnp.nan_to_num(grad_Z_f(theta, X, y, lbd_v), nan=lbd_v)
        hess_Z = jnp.nan_to_num(hess_Z_f(theta, X, y, lbd_v), nan=lbd_v)

        grad_minus = f_grad - grad_Z
        hess_minus = f_hess - hess_Z

        theta_cv = prox_lasso(
            theta_cv
            - alpha_t * grad_minus
            - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta)),
            lbd_v,
        )
        iacv_end = time.time()

        iacv_sparse_start = time.time()

        # def iacv_operation(X_s, y, theta_s, s):
        #    f_grad_s = jnp.nan_to_num(nabla_F_s(theta_s, X_s, y, lbd_v), nan=lbd_v)
        #    f_hess_s = jnp.nan_to_num(hess_F_s(theta_s, X_s, y, lbd_v), nan=lbd_v)

        #    grad_Z_s = jnp.nan_to_num(grad_Z_f_s(theta_s, X_s, y, lbd_v), nan=lbd_v)
        #    hess_Z_s = jnp.nan_to_num(hess_Z_f_s(theta_s, X_s, y, lbd_v), nan=lbd_v)

        #    grad_minus = f_grad_s - grad_Z_s
        #    hess_minus = f_hess_s - hess_Z_s

        #    return (grad_minus, hess_minus)
        #    # theta_cv_sparse[:, s] = prox_lasso(
        #    #    theta_cv_sparse[:, s]
        #    #    - alpha_t * grad_minus
        #    #    - alpha_t * vmap_matmul(hess_minus, (theta_cv_sparse[:, s] - theta[s])),
        #    #    lbd_v,
        #    # )

        # iacv_sparse_op = vmap(iacv_operation, in_axes=(1, None, 1, 0))
        # iacv_update = iacv_sparse_op(
        #    X[:, support].reshape(n, len(support[0])),
        #    y,
        #    theta[support].reshape(1, -1),
        #    support[0],
        # )

        # print(theta_cv_sparse[:, 2].shape)
        # print(iacv_update[0][2, :].shape)
        # print(iacv_update[1][2, :].shape)
        # print(iacv_update[1][2, :].shape)

        # for s in support[0]:
        #    theta_cv_sparse[:, s] = prox_lasso(
        #        theta_cv_sparse[:, s]
        #        - alpha_t * iacv_update[0][s, :]
        #        - alpha_t * iacv_update[1][s, :] * (theta_cv_sparse[:, s] - theta[s]),
        #        lbd_v,
        #    )

        # for s in support:

        # set the zero'd out submatrices
        X_s = np.zeros(X.shape)
        X_s[:, support] = X[:, support]

        theta_s = np.zeros(theta.shape)
        theta_s[support] = theta[support]

        f_grad_s = jnp.nan_to_num(nabla_F(theta_s, X_s, y, lbd_v), nan=lbd_v)
        f_hess_s = jnp.nan_to_num(hess_F(theta_s, X_s, y, lbd_v), nan=lbd_v)

        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)
        hess_Z_s = jnp.nan_to_num(hess_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)

        grad_minus = f_grad_s - grad_Z_s
        hess_minus = f_hess_s - hess_Z_s

        theta_cv_sparse = prox_lasso(
            theta_cv_sparse
            - alpha_t * grad_minus
            - alpha_t * vmap_matmul(hess_minus, (theta_cv_sparse - theta_s)),
            lbd_v,
        )
        iacv_sparse_end = time.time()

        # NS block
        ns_start = time.time()
        # for s in support:
        #    f_hess_s = jnp.nan_to_num(hess_F(theta[s], X[:, s], y, lbd_v), nan=lbd_v)

        #    grad_Z_s = jnp.nan_to_num(grad_Z_f(theta[s], X[:, s], y, lbd_v), nan=lbd_v)
        #    hess_Z_s = jnp.nan_to_num(hess_Z_f(theta[s], X[:, s], y, lbd_v), nan=lbd_v)

        #    hess_minus_s = f_hess_s - hess_Z_s
        #    theta_ns[:, s] = theta[s] + jnp.nan_to_num(
        #        vmap_matmul(jnp.linalg.inv(hess_minus_s), grad_Z_s), nan=lbd_v
        #    )
        ns_end = time.time()

        # IJ block
        ij_start = time.time()

        X_s = np.zeros(X.shape)
        X_s[:, support] = X[:, support]

        theta_s = np.zeros(theta.shape)
        theta_s[support] = theta[support]

        f_hess_s = jnp.nan_to_num(hess_F(theta_s, X_s, y, lbd_v), nan=lbd_v)
        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)

        theta_ij = theta_s + jnp.nan_to_num(
            jit(vmap(jnp.matmul, in_axes=(None, 0)))(
                jnp.linalg.inv(f_hess_s), grad_Z_s
            ),
            nan=lbd_v,
        )
        ij_end = time.time()

        # true block
        true_start = time.time()
        for i in range(n):
            theta_true[i] = prox_lasso(
                theta_true[i]
                - alpha
                * jnp.nan_to_num(
                    nabla_F(
                        theta_true[i],
                        X[mask[i, :]],
                        y[mask[i, :]],
                        lbd=lbd_v,
                    ),
                    nan=lbd_v,
                ),
                lbd_v,
            )
        true_end = time.time()

        # actually update theta
        theta = prox_lasso(theta - alpha * f_grad, lbd_v)

        true_stack = jnp.stack(theta_true)
        for k, v in zip(
            err_approx.keys(), [theta_cv, theta_cv_sparse, theta_ns, theta_ij, theta]
        ):
            err_approx[k][t] = jnp.mean(jnp.linalg.norm(v - true_stack, 2, axis=1))

        # track runtimes
        for k, (t1, t2) in zip(
            runtime.keys(),
            [
                (iacv_start, iacv_end),
                (iacv_sparse_start, iacv_sparse_end),
                (ns_start, ns_end),
                (ij_start, ij_end),
                (true_start, true_end),
            ],
        ):
            runtime[k][t] = runtime[k][t - 1] + t2 - t1 if t > 0 else 0

    params = {
        "IACV": theta_cv,
        "IACV sparse": theta_cv_sparse,
        "IJ": theta_ij,
        "NS": theta_ns,
        "true": theta_true,
    }

    return err_approx, runtime, supports, params
