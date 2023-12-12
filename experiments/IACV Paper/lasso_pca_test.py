#!/usr/bin/env python3

import numpy as np
from jax import vmap, jacrev, jacfwd, grad, jit, pmap, profiler
from sklearn.decomposition import PCA
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


if "__main__":
    n = 100
    p = 200
    n_iter = 1200
    lbd_v = 3e-3

    X, theta_star, y = sample_from_logreg_first_5(p=p, n=n)

    X_pca = PCA(0.6).fit_transform(X)

    true_support = np.where(theta_star != 0)[0]

    if lbd_v is None:
        lbd_v = 1e-6 * n

    theta = jnp.ones((p, 1))
    alpha = 0.5 / n
    alpha_t = alpha

    theta_pca = jnp.ones((X_pca.shape[1], 1))

    nabla_F = jit(grad(F_mod))

    for t in range(0, n_iter):
        support = np.where(theta != 0)[0]

        print(f"iter: {t} | support size: {len(support)}")

        f_grad = jnp.nan_to_num(nabla_F(theta, X, y, lbd_v), nan=lbd_v)
        # actually update theta
        theta = prox_lasso(theta - alpha * f_grad, lbd_v)

        f_grad_pca = jnp.nan_to_num(nabla_F(theta_pca, X_pca, y, lbd_v), nan=lbd_v)
        theta_pca = prox_lasso(theta_pca - alpha * f_grad_pca, lbd_v)

    print(theta.flatten())
    print(theta_pca.flatten())

    print("losses: ")
    print(f"reg: {jnp.linalg.norm(X @ theta - y, 2)}")
    print(f"pca: {jnp.linalg.norm(X_pca @ theta_pca - y, 2)}")
