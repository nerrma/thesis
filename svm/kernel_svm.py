#!/usr/bin/env python3
import numpy as np
from jax import vmap, jacrev, jacfwd, grad, jit
from sklearn.metrics import accuracy_score
import jax.numpy as jnp


class SVM_smooth_kernel:
    def __init__(self, sigma: np.float64, lbd: np.float64, kernel=None, **kernel_args):
        self.sigma_ = sigma
        self.lbd_ = lbd

        self.n = 1
        self.p = 1

        self.kernel_ = lambda x, y: x @ y.T
        if kernel:
            self.kernel_ = kernel

        self.kernel_args_ = kernel_args
        self.u_ = np.zeros(0)
        self.gram_ = np.zeros(0)

    def Phi_m(self, v):
        return (1 + v / np.sqrt(1 + v**2)) / 2

    def Phi_m_prime(self, v):
        return 1 / 2 * 1 / (np.sqrt(v**2 + 1) ** 3)

    def phi_m(self, v):
        return 1 / (2 * np.sqrt(1 + v**2))

    def Phi_m_jax(self, v):
        return (1 + v / jnp.sqrt(1 + v**2)) / 2

    def Phi_m_prime_jax(self, v):
        return 1 / 2 * 1 / (jnp.sqrt(v**2 + 1) ** 3)

    def phi_m_jax(self, v):
        return 1 / (2 * jnp.sqrt(1 + v**2))

    def Psi_m(self, alpha, sigma):
        return (
            self.Phi_m_jax((1 - alpha) / sigma) * (1 - alpha)
            + self.phi_m_jax((1 - alpha) / sigma) * sigma
        )

    def nabla_fgd_(
        self,
        u: np.ndarray,
        gram: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return (
            lbd * gram @ u
            - 1 / self.n * self.Phi_m((1 - y * (gram @ u)) / sigma) * y @ gram
        )

    def hess_fgd_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (gram @ u)) / sigma) / sigma
        hess = lbd * gram + 1 / self.n * gram.T * d @ gram

        return hess

    def loss(self, u, gram, y, sigma):
        return jnp.mean(self.Psi_m(y * (gram @ u), sigma))

    def SSVM_objective(self, u, gram, y, sigma, lbd):
        return 1 / 2 * (u.T @ gram @ u) * lbd + self.loss(u, gram, y, sigma)

    def fit_gd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-3,
        n_iter=1000,
    ):
        self.gram_ = self.kernel_(X, X, **self.kernel_args_)

        for t in range(n_iter):
            f_grad = self.nabla_fgd_(self.u_, self.gram_, y, self.sigma_, self.lbd_)
            hess = self.hess_fgd_(self.u_, self.gram_, y, self.sigma_, self.lbd_)

            self.u_ = self.u_ - eta * f_grad
            print(
                f"iter {t} | f_grad {np.linalg.norm(f_grad)} | objective {self.SSVM_objective(self.u_, self.gram_, y, self.sigma_, self.lbd_)}"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-4,
        n_iter=1000,
        thresh=1e-8,
        init_u=None,
        **kwargs,
    ):
        self.n = X.shape[0]
        self.p = X.shape[1]

        self.u_ = np.zeros(self.n)
        if init_u is not None:
            self.u_ = init_u

        self.fit_gd_(
            X,
            y,
            eta=eta,
            n_iter=n_iter,
            **kwargs,
        )

    def predict(self, X: np.ndarray):
        return np.sign(self.gram_ @ self.u_)
