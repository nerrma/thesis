#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from functools import partial
import jax.numpy as jnp
from sampler import sample_from_logreg


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

    nabla_F = jit(grad(F_mod))
    hess_F = jit(jacfwd(jacrev(F_mod)))

    grad_Z_f = jit(vmap(nabla_F, in_axes=(None, 0, 0, None)))
    hess_Z_f = jit(vmap(hess_F, in_axes=(None, 0, 0, None)))
    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

    for t in range(0, n_iter):
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

        theta_ns = theta + jnp.nan_to_num(
            vmap_matmul(jnp.linalg.inv(hess_minus), grad_Z), nan=lbd_v
        )
        theta_ij = theta + jnp.nan_to_num(
            jit(vmap(jnp.matmul, in_axes=(None, 0)))(jnp.linalg.inv(f_hess), grad_Z),
            nan=lbd_v,
        )

        for i in range(n):
            theta_true[i] = theta_true[i] - alpha * jnp.nan_to_num(
                nabla_F(
                    theta_true[i],
                    np.delete(X, (i), axis=0),
                    np.delete(y, (i), axis=0),
                    lbd=lbd_v,
                ),
                nan=lbd_v,
            )

        # actually update theta
        theta = theta - alpha * f_grad

        true_stack = jnp.stack(theta_true)
        err_approx["IACV"][t] = jnp.mean(
            jnp.linalg.norm(theta_cv - true_stack, 2, axis=1)
        )
        err_approx["NS"][t] = jnp.mean(
            jnp.linalg.norm(theta_ns - true_stack, 2, axis=1)
        )
        err_approx["IJ"][t] = jnp.mean(
            jnp.linalg.norm(theta_ij - true_stack, 2, axis=1)
        )
        err_approx["hat"][t] = jnp.mean(jnp.linalg.norm(theta - true_stack, 2, axis=1))

    return err_approx
