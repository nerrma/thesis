#!/usr/bin/env python3
import numpy as np
import time
from jax import vmap, jit
import jax.numpy as jnp


# An SVM class which learns using a Smooth Hinge loss
class SVM_smooth:
    def __init__(self, sigma: np.float64, lbd: np.float64):
        self.sigma_ = sigma
        self.lbd_ = lbd

        self.n = 1
        self.p = 1

        self.weights_ = np.zeros(0)
        self.grads_ = []
        self.hess_ = []

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

    def nabla_fgd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * self.Phi_m((1 - y * (X @ w)) / sigma) * y @ X

    def nabla_fgd_single_(
        self,
        X: np.ndarray,
        y: np.float64,
        w: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * self.Phi_m_jax((1 - y * (X * w)) / sigma) * y * X

    def hess_fgd_(self, X, y, w, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (X @ w)) / sigma) / sigma
        # D = np.eye(d.shape[0])
        # for i in range(d.shape[0]):
        #    D[i, i] = d[i]

        # hess = lbd * np.eye(self.p) + 1 / self.n * X.T @ D @ X
        hess = lbd * np.eye(self.p) + 1 / self.n * X.T * d @ X
        return hess

    def hess_fgd_single_(self, X, y, w, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X * w)) / sigma) / sigma
        # D = jnp.eye(self.n)
        # jnp.fill_diagonal(D, d)
        # diag_elements = jnp.diag_indices_from(D)
        # D = D.at[diag_elements].set(d)
        # for i in range(self.n):
        #    D[i, i] = d.at[i]

        hess = lbd * np.eye(self.p) + 1 / self.n * X.T * d * X
        return hess

    def fit_gd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-4,
        cv=False,
        approx_cv=False,
        n_iter=1000,
        thresh=1e-8,
        log_iacv=False,
        log_iter=False,
        save_grads=False,
        save_hess=False,
    ):
        alpha_t = eta

        grad_Z_f = jit(vmap(self.nabla_fgd_single_, in_axes=(0, 0, None, None, None)))
        hess_Z_f = jit(vmap(self.hess_fgd_single_, in_axes=(0, 0, None, None, None)))
        vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

        for t in range(n_iter):
            f_grad = self.nabla_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)

            if np.linalg.norm(f_grad) < thresh:
                # print(f"stopping early at iteration {t}")
                break

            if approx_cv == True:
                # vectorised per sample hessian
                start = time.time()
                f_hess = self.hess_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)

                grad_per_sample = grad_Z_f(X, y, self.weights_, self.sigma_, self.lbd_)
                hess_per_sample = hess_Z_f(X, y, self.weights_, self.sigma_, self.lbd_)

                # per sample gradient and hessian difference
                hess_minus_i = f_hess - hess_per_sample
                grad_minus_i = f_grad - grad_per_sample

                print(
                    f"hess: {np.linalg.norm(f_hess):.8f} | per sample hess: {np.linalg.norm(hess_per_sample):.8f} | per sample grad: {np.linalg.norm(grad_per_sample):.8f}"
                )

                print(
                    f"hess minus: {np.linalg.norm(hess_minus_i)} | grad minus i: {np.linalg.norm(grad_minus_i)}"
                )

                # if np.linalg.norm(f_hess) > 100:
                #    print(np.linalg.norm(hess_per_sample))
                #    print(f_hess)
                #    break
                # print(
                #    f"hess * diff {np.linalg.norm(hess_minus_i[0]) * (self.loo_iacv_[0] - self.weights_)} and {np.linalg.norm(grad_minus_i[0])}"
                # )

                self.loo_iacv_ = (
                    self.loo_iacv_
                    - alpha_t * grad_minus_i
                    - alpha_t
                    * vmap_matmul(hess_minus_i, (self.loo_iacv_ - self.weights_))
                )

                end = time.time()

            if cv == True:
                start = time.time()
                for i in range(self.n):
                    X_temp = np.delete(X, (i), axis=0)
                    y_temp = np.delete(y, (i), axis=0)
                    self.loo_true_[i] = self.loo_true_[i] - eta * self.nabla_fgd_(
                        X_temp, y_temp, self.weights_, self.sigma_, self.lbd_
                    )
                end = time.time()

            if log_iter == True:
                print(f"iter {t} | grad {np.linalg.norm(f_grad):.5f} ", end="")

            if log_iacv == True:
                print(
                    f"IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1)):.8f}"
                )
            elif log_iter:
                print("\n", end="")

            self.weights_ = self.weights_ - eta * f_grad

            if save_grads == True:
                self.grads_.append(f_grad)

            if save_hess == True:
                self.hess_.append(f_hess)

            # normalise?
            # self.weights_ /= np.linalg.norm(self.weights_)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-4,
        cv=True,
        approx_cv=True,
        n_iter=1000,
        thresh=1e-8,
        init_w=None,
        **kwargs,
    ):
        self.n = X.shape[0]
        self.p = X.shape[1]

        self.weights_ = np.zeros(self.p)
        if init_w is not None:
            self.weights_ = init_w

        self.loo_true_ = np.zeros((self.n, self.p))
        self.loo_iacv_ = np.zeros((self.n, self.p))

        self.fit_gd_(
            X,
            y,
            eta=eta,
            cv=cv,
            approx_cv=approx_cv,
            n_iter=n_iter,
            thresh=thresh,
            **kwargs,
        )

    def predict(self, X: np.ndarray):
        return np.sign(X @ self.weights_)
