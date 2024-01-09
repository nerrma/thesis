#!/usr/bin/env python3
import numpy as np
import time
from jax import vmap, jacrev, jacfwd, grad, jit
from scipy import linalg as sc_linalg
from sklearn.preprocessing import normalize
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
        self.cond_nums_ = []
        self.err_approx_ = {"IACV": [], "baseline": []}
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

    def phi_m_jax(self, v):
        return 1 / (2 * jnp.sqrt(1 + v**2))

    def Psi_m(self, alpha, sigma):
        return (
            self.Phi_m_jax((1 - alpha) / sigma) * (1 - alpha)
            + self.phi_m_jax((1 - alpha) / sigma) * sigma
        )

    def nabla_fgd_(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * self.Phi_m((1 - y * (X @ w)) / sigma) * y @ X

    def nabla_fgd_single_(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.float64,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * self.Phi_m_jax((1 - y * (X * w)) / sigma) * y * X

    def hess_fgd_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (X @ w)) / sigma) / sigma
        # D = np.eye(d.shape[0])
        # for i in range(d.shape[0]):
        #    D[i, i] = d[i]

        # hess = lbd * np.eye(self.p) + 1 / self.n * X.T @ D @ X
        hess = lbd * np.eye(self.p) + 1 / self.n * X.T * d @ X
        return hess

    def hess_fgd_single_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X @ w)) / sigma) / sigma
        # D = jnp.eye(self.n)
        # jnp.fill_diagonal(D, d)
        # diag_elements = jnp.diag_indices_from(D)
        # D = D.at[diag_elements].set(d)
        # for i in range(self.n):
        #    D[i, i] = d.at[i]

        # hess = lbd * np.eye(self.p) + 1 / self.n * d * X.T * X
        # X = X.reshape(-1, 1)
        hess = lbd * np.eye(self.p) + 1 / self.n * d * X * X
        # hess = lbd * np.eye(self.p) + 1 / self.n * X.T * d @ X
        return hess

    def SSVM_objective(self, w, X, y, sigma, lbd):
        return 1 / 2 * jnp.linalg.norm(w) * lbd + jnp.mean(
            self.Psi_m(y * (X @ w), sigma)
        )

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
        log_cond_number=False,
        save_grads=False,
        save_hess=False,
        save_cond_nums=False,
        save_err_approx=False,
        use_jax_grad=False,
        normalise=False,
    ):
        alpha_t = eta

        grad_Z_f = jit(vmap(self.nabla_fgd_single_, in_axes=(None, 0, 0, None, None)))
        hess_Z_f = jit(vmap(self.hess_fgd_single_, in_axes=(None, 0, 0, None, None)))

        # define jax grad variables
        jax_grad = jit(grad(self.SSVM_objective))
        jax_hess = jit(jacfwd(jacrev(self.SSVM_objective)))

        if use_jax_grad:
            grad_Z_f = jit(vmap(jax_grad, in_axes=(None, 0, 0, None, None)))
            hess_Z_f = jit(vmap(jax_hess, in_axes=(None, 0, 0, None, None)))

        vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))

        for t in range(n_iter):
            if use_jax_grad:
                f_grad = jax_grad(self.weights_, X, y, self.sigma_, self.lbd_)
            else:
                f_grad = self.nabla_fgd_(self.weights_, X, y, self.sigma_, self.lbd_)

            if np.linalg.norm(f_grad) < thresh:
                print(f"stopping early at iteration {t}")
                break

            if approx_cv == True:
                # vectorised per sample hessian
                start = time.time()
                if use_jax_grad:
                    f_hess = jax_hess(self.weights_, X, y, self.sigma_, self.lbd_)
                else:
                    f_hess = self.hess_fgd_(self.weights_, X, y, self.sigma_, self.lbd_)

                grad_per_sample = grad_Z_f(self.weights_, X, y, self.sigma_, self.lbd_)
                hess_per_sample = hess_Z_f(self.weights_, X, y, self.sigma_, self.lbd_)

                # per sample gradient and hessian difference
                hess_minus_i = f_hess - hess_per_sample
                grad_minus_i = f_grad - grad_per_sample

                if log_cond_number or save_cond_nums:
                    cond_num = np.linalg.cond(hess_minus_i)
                    if save_cond_nums:
                        self.cond_nums_.append(cond_num)

                    if log_cond_number:
                        print(
                            f"hessian condition number {np.mean(np.linalg.cond(f_hess))}"
                        )
                        print(
                            f"mean hessian condition number {np.mean(cond_num)} | min hessian condition number {np.min(cond_num, axis=0)} | max hessian condition number {np.max(cond_num, axis=0)}"
                        )

                self.loo_iacv_ = (
                    self.loo_iacv_
                    - alpha_t * grad_minus_i
                    - alpha_t
                    * vmap_matmul(hess_minus_i, (self.loo_iacv_ - self.weights_))
                )
                if normalise:
                    self.loo_iacv_ = normalize(self.loo_iacv_, axis=1)

                end = time.time()

            if cv == True:
                start = time.time()
                for i in range(self.n):
                    X_temp = np.delete(X, (i), axis=0)
                    y_temp = np.delete(y, (i), axis=0)
                    self.loo_true_[i] = self.loo_true_[i] - eta * self.nabla_fgd_(
                        self.weights_, X_temp, y_temp, self.sigma_, self.lbd_
                    )
                end = time.time()
                if normalise:
                    self.loo_true_ = normalize(self.loo_true_, axis=1)

            if log_iter == True:
                print(
                    f"iter {t} | grad {np.linalg.norm(f_grad):.5f} | objective {self.SSVM_objective(self.weights_, X, y, self.sigma_, self.lbd_):.5f} ",
                    end="",
                )

            if log_iacv == True:
                print(
                    f"IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1)):.8f}"
                )
            elif log_iter:
                print("\n", end="")

            if save_err_approx:
                self.err_approx_["IACV"].append(
                    np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1))
                )

                self.err_approx_["baseline"].append(
                    np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1))
                )

            self.weights_ = self.weights_ - eta * f_grad

            if save_grads == True:
                self.grads_.append(f_grad)

            if save_hess == True:
                self.hess_.append(f_hess)

            # normalise?
            if normalise:
                self.weights_ /= np.linalg.norm(self.weights_)

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
