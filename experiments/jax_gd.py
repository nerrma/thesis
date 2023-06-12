#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from jax import vmap, jacrev, jacfwd, grad, jit, pmap
import jax.numpy as jnp
from sampler import sample_from_logreg


def l(X, y, theta):
    return -y * (X @ theta) + jnp.log(1 + jnp.exp(X @ theta))


def pi(theta):
    return jnp.linalg.norm(theta, 2)


@jit
def F_mod(theta, X, y, lbd):
    return jnp.sum(l(X, y, theta)) + lbd * pi(theta)


@jit
def update(
    theta,
    theta_cv,
    theta_ns,
    theta_ij,
    grad_Z,
    grad_minus,
    hess_minus,
    f_hess,
    alpha_t=1e4,
):
    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

    theta_cv = (
        theta_cv
        - alpha_t * grad_minus
        - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta))
    )

    theta_ns = theta + vmap_matmul(jnp.linalg.pinv(hess_minus), grad_Z)
    theta_ij = theta + vmap(jnp.matmul, in_axes=(None, 0))(
        jnp.linalg.pinv(f_hess), grad_Z
    )

    return (theta_cv, theta_ns, theta_ij)


def run_sim(n, p, n_iter=250):
    X, _, y = sample_from_logreg(p=p, n=n)

    lbd_v = 1e-6 * n

    theta = jnp.ones(p)
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = jnp.ones((n, p))
    theta_true = [jnp.ones(p)] * n
    theta_ns = jnp.ones((n, p))
    theta_ij = jnp.ones((n, p))

    err_approx = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    nabla_F = jit(grad(F_mod))
    hess_F = jit(jacfwd(jacrev(F_mod)))

    mask = ~jnp.diag(jnp.ones(n, dtype=bool))
    grad_Z_f = jit(vmap(nabla_F, in_axes=(None, 0, 0, None)))
    hess_Z_f = jit(vmap(hess_F, in_axes=(None, 0, 0, None)))
    vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

    for t in range(0, n_iter):
        f_grad = nabla_F(theta, X, y, lbd_v)
        f_hess = jnp.nan_to_num(hess_F(theta, X, y, lbd_v))

        grad_Z = grad_Z_f(theta, X, y, lbd_v)
        hess_Z = hess_Z_f(theta, X, y, lbd_v)

        grad_minus = f_grad - grad_Z
        hess_minus = f_hess - hess_Z

        # theta_cv, theta_ns, theta_ij = update(
        #    theta,
        #    theta_cv,
        #    theta_ns,
        #    theta_ij,
        #    grad_Z,
        #    grad_minus,
        #    hess_minus,
        #    f_hess,
        #    alpha_t=alpha_t,
        # )

        theta_cv = (
            theta_cv
            - alpha_t * grad_minus
            - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta))
        )

        theta_ns = theta + vmap_matmul(jnp.linalg.pinv(hess_minus), grad_Z)
        theta_ij = theta + vmap(jnp.matmul, in_axes=(None, 0))(
            jnp.linalg.pinv(f_hess), grad_Z
        )

        for i in range(n):
            theta_true[i] = theta_true[i] - alpha * nabla_F(
                theta_true[i], X[mask[i, :]], y[mask[i, :]], lbd_v
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
