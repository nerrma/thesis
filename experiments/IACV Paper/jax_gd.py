#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from functools import partial
import jax.numpy as jnp
import time
from sampler import sample_from_logreg

import jax

jax.config.update("jax_platform_name", "cpu")


def l(X, y, theta):
    return -y * (X @ theta) + jnp.log(1 + jnp.exp(X @ theta))


def pi(theta):
    return jnp.linalg.norm(theta, 2)


def F_mod(theta, X, y, lbd):
    return jnp.sum(l(X, y, theta)) + lbd * pi(theta)


def run_sim(n, p, n_iter=250):
    X, _, y = sample_from_logreg(p=p, n=n)

    lbd_v = 1e-6 * n

    theta = jnp.zeros(p)
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = jnp.zeros((n, p))
    theta_true = [jnp.zeros(p)] * n
    theta_ns = jnp.zeros((n, p))
    theta_ij = jnp.zeros((n, p))

    err_approx = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    cv_err = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    runtime = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "true": np.zeros(n_iter),
    }

    nabla_F = jit(grad(F_mod))
    hess_F = jit(jacfwd(jacrev(F_mod)))

    grad_Z_f = jit(vmap(nabla_F, in_axes=(None, 0, 0, None)))
    hess_Z_f = jit(vmap(hess_F, in_axes=(None, 0, 0, None)))
    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))
    one_axis_matmul = jit(vmap(jnp.matmul, in_axes=(None, 0)))

    mask = ~np.eye(n, dtype=bool)
    for t in range(0, n_iter):
        # IACV BLOCK
        iacv_start = time.time()
        f_grad = jnp.nan_to_num(nabla_F(theta, X, y, lbd_v), nan=lbd_v)
        f_hess = jnp.nan_to_num(hess_F(theta, X, y, lbd_v), nan=lbd_v)

        grad_Z = jnp.nan_to_num(grad_Z_f(theta, X, y, lbd_v), nan=lbd_v)
        hess_Z = jnp.nan_to_num(hess_Z_f(theta, X, y, lbd_v), nan=lbd_v)

        grad_minus = f_grad - grad_Z
        hess_minus = f_hess - hess_Z

        theta_cv = (
            theta_cv
            - alpha_t * grad_minus
            - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta))
        )

        iacv_end = time.time()

        # NS BLOCK
        ns_start = time.time()
        f_hess = jnp.nan_to_num(hess_F(theta, X, y, lbd_v), nan=lbd_v)

        grad_Z = jnp.nan_to_num(grad_Z_f(theta, X, y, lbd_v), nan=lbd_v)
        hess_Z = jnp.nan_to_num(hess_Z_f(theta, X, y, lbd_v), nan=lbd_v)

        hess_minus = f_hess - hess_Z
        theta_ns = theta + jnp.nan_to_num(
            vmap_matmul(jnp.linalg.inv(hess_minus), grad_Z), nan=lbd_v
        )
        ns_end = time.time()

        # IJ BLOCK
        ij_start = time.time()
        f_hess = jnp.nan_to_num(hess_F(theta, X, y, lbd_v), nan=lbd_v)
        grad_Z = jnp.nan_to_num(grad_Z_f(theta, X, y, lbd_v), nan=lbd_v)
        f_hess_inv = jnp.linalg.inv(f_hess)
        theta_ij = theta + jnp.nan_to_num(
            one_axis_matmul(f_hess_inv, grad_Z),
            nan=lbd_v,
        )
        ij_end = time.time()

        true_cv_start = time.time()
        for i in range(n):
            theta_true[i] = theta_true[i] - alpha * jnp.nan_to_num(
                nabla_F(
                    theta_true[i],
                    X[mask[i, :]],
                    y[mask[i, :]],
                    lbd=lbd_v,
                ),
                nan=lbd_v,
            )

        true_cv_end = time.time()

        # actually update theta
        theta = theta - alpha * f_grad

        true_stack = jnp.stack(theta_true)

        for k, v in zip(err_approx.keys(), [theta_cv, theta_ns, theta_ij, theta]):
            err_approx[k][t] = jnp.mean(jnp.linalg.norm(v - true_stack, 2, axis=1))

        loss_vmap = jit(vmap(l, in_axes=(0, 0, 0)))
        for k, v in zip(cv_err.keys(), [theta_cv, theta_ns, theta_ij, theta]):
            lv = loss_vmap if len(v.shape) > 1 else l
            cv_err[k][t] = jnp.abs(loss_vmap(X, y, true_stack) - lv(X, y, v)).mean()

        for k, (t1, t2) in zip(
            runtime.keys(),
            [
                (iacv_start, iacv_end),
                (ns_start, ns_end),
                (ij_start, ij_end),
                (true_cv_start, true_cv_end),
            ],
        ):
            # runtime[k][t] = runtime[k][t - 1] + t2 - t1 if t > 0 else t2 - t1
            runtime[k][t] = runtime[k][t - 1] + t2 - t1 if t > 0 else 0

        print(
            f"IACV: {err_approx['IACV'][t]} | baseline: {err_approx['hat'][t]} | NS: {err_approx['NS'][t]} | IJ {err_approx['IJ'][t]}"
        )

    return err_approx, cv_err, runtime
