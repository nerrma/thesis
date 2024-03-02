#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from functools import partial
import jax.numpy as jnp
from jaxopt.prox import prox_lasso
from sampler import sample_from_logreg_first_5

import jax

jax.config.update("jax_platform_name", "cpu")


def l(X, y, theta):
    return -y * (X @ theta) + jnp.log(1 + jnp.exp(X @ theta))


def l_s(X, y, theta):
    return -y * jnp.sum(X * theta) + jnp.log(1 + jnp.exp(jnp.sum(X * theta)))


def pi(theta):
    return jnp.linalg.norm(theta, 1)


def F_mod(theta, X, y, lbd):
    return jnp.sum(l(X, y, theta))


def F_mod_s(theta, X, y, lbd):
    return jnp.sum(l_s(X, y, theta)) + lbd * pi(theta)


def run_sim(n, p, n_iter=250, lbd_v=None):
    X, theta_star, y = sample_from_logreg_first_5(p=p, n=n)

    true_support = np.where(theta_star != 0)[0]

    if lbd_v is None:
        lbd_v = 1e-6 * n

    theta = jnp.ones(p)
    alpha = 0.5 / n
    alpha_t = alpha

    # theta_cv = jnp.ones((n, p))
    theta_cv = np.ones((n, p))
    theta_cv_sparse = np.ones((n, p))
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

    # non JIT variants
    nabla_F_no_jit = grad(F_mod)
    hess_F_no_jit = jacfwd(jacrev(F_mod))

    grad_Z_f_no_jit = vmap(nabla_F, in_axes=(None, 0, 0, None))
    hess_Z_f_no_jit = vmap(hess_F, in_axes=(None, 0, 0, None))

    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))
    vmap_matmul_no_jit = vmap(jnp.matmul, in_axes=(0, 0))
    one_axis_matmul = jit(vmap(jnp.matmul, in_axes=(None, 0)))

    mask = ~np.eye(n, dtype=bool)
    for t in range(0, n_iter):
        support = np.where(theta != 0)[0]
        # not_support = np.where(np.isclose(theta, 0))[0]
        supports.append(support)

        print(f"iter: {t} | support size: {len(support)}")

        X_s = np.zeros(X.shape)
        X_s[:, support] = X[:, support]

        theta_s = np.zeros(theta.shape)
        theta_s[support] = theta[support]

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

        # IACV on the support
        iacv_sparse_start = time.time()

        # set the zero'd out submatrices
        # X_s = jnp.compress(supp_bool, X, axis=1)
        # theta_s = jnp.compress(supp_bool, theta)
        # theta_cv_sparse_compress = jnp.compress(supp_bool, theta_cv_sparse, axis=1)

        f_grad_s = jnp.nan_to_num(nabla_F(theta_s, X_s, y, lbd_v), nan=lbd_v)
        f_hess_s = jnp.nan_to_num(hess_F(theta_s, X_s, y, lbd_v), nan=lbd_v)

        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)
        hess_Z_s = jnp.nan_to_num(hess_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)

        grad_minus_s = f_grad_s - grad_Z_s
        hess_minus_s = f_hess_s - hess_Z_s

        # theta_cv_sparse_compress = prox_lasso(
        #    theta_cv_sparse_compress
        #    - alpha_t * grad_minus
        #    - alpha_t
        #    * vmap_matmul_no_jit(hess_minus, (theta_cv_sparse_compress - theta_s)),
        #    lbd_v,
        # )

        theta_cv_sparse = prox_lasso(
            theta_cv_sparse
            - alpha_t * grad_minus_s
            - alpha_t * vmap_matmul(hess_minus_s, (theta_cv_sparse - theta_s)),
            lbd_v,
        )

        # ensure sparse matrix is in desired form
        # np.place(theta_cv_sparse, [[theta != 0] * n], theta_cv_sparse_compress)
        # print(theta_cv_sparse[:, support].shape)
        # print(theta_cv_sparse_compress.shape)
        # theta_cv_sparse[:, support] = theta_cv_sparse_compress
        iacv_sparse_end = time.time()

        # NS block
        ns_start = time.time()
        f_hess_s = jnp.nan_to_num(hess_F(theta_s, X_s, y, lbd_v), nan=lbd_v)

        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)
        hess_Z_s = jnp.nan_to_num(hess_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)

        hess_minus_s = f_hess_s - hess_Z_s
        theta_ns = theta_s + jnp.nan_to_num(
            vmap_matmul(jnp.linalg.inv(hess_minus_s), grad_Z_s), nan=lbd_v
        )
        ns_end = time.time()

        # IJ block
        ij_start = time.time()

        f_hess_s = jnp.nan_to_num(hess_F(theta_s, X_s, y, lbd_v), nan=lbd_v)
        grad_Z_s = jnp.nan_to_num(grad_Z_f(theta_s, X_s, y, lbd_v), nan=lbd_v)

        theta_ij = theta_s + jnp.nan_to_num(
            one_axis_matmul(jit(jnp.linalg.inv)(f_hess_s), grad_Z_s),
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
            err_approx.keys(),
            [theta_cv, theta_cv_sparse, theta_ns, theta_ij, theta],
        ):
            err_approx[k][t] = jnp.mean(jnp.linalg.norm(v - true_stack, 2, axis=1))

        true_LOO = F_mod(true_stack.T, X, y, lbd_v)
        for k, v in list(
            [
                ("NS", theta_ns),
                ("IJ", theta_ij),
                ("hat", theta),
                ("IACV", theta_cv),
                ("IACV sparse", theta_cv_sparse),
            ],
        ):
            err_loo[k][t] = min(
                1, (jnp.mean(jnp.abs(F_mod(v.T, X, y, lbd_v) - true_LOO)) / true_LOO)
            )

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
        "star": theta_star,
    }

    return err_approx, err_loo, runtime, supports, params


# if "__main__":
#    n = 20
#    p = 50
#    special_lambda = 0.25 * np.sqrt(np.log(p) / n)
#    print(f"special_lambda is {special_lambda}")
#    err_approx, _, runtime, supports, params = run_sim(
#        n=n, p=p, n_iter=100, lbd_v=special_lambda
#    )
#
#    print(err_approx)
