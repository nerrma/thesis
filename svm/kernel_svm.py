#!/usr/bin/env python3
import numpy as np
from jax import vmap, jacrev, jacfwd, grad, jit
from sklearn.metrics import accuracy_score
import jax.numpy as jnp

from iacv import IACV
from true_cv import TrueCV


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
        self.factor = self.n

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

    def nabla_fgd_no_reg_no_factor_(
        self,
        u: np.ndarray,
        gram: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return -self.Phi_m((1 - y * (gram @ u)) / sigma) * y @ gram

    def nabla_fgd_single_(
        self,
        u: np.ndarray,
        gram: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return (
            lbd * gram @ u
            - 1 / self.n * self.Phi_m_jax((1 - y * (gram @ u)) / sigma) * y @ gram
        )

    def nabla_fgd_single_no_factor_(
        self,
        u: np.ndarray,
        gram: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return -self.Phi_m_jax((1 - y * (gram * u)) / sigma) * y * gram

    def hess_fgd_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (gram @ u)) / sigma) / sigma
        hess = lbd * gram + 1 / self.n * gram.T * d @ gram

        return hess

    def hess_fgd_no_reg_no_factor_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (gram @ u)) / sigma) / sigma
        hess = np.zeros((self.n, self.n)) + gram.T * d @ gram

        return hess

    def hess_fgd_single_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (gram @ u)) / sigma) / sigma
        hess = lbd * gram + 1 / self.n * gram.T * d @ gram

        return hess

    def hess_fgd_single_no_factor_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (gram @ u)) / sigma) / sigma
        hess = np.zeros((self.n, self.n)) + gram.T * d @ gram

        return hess

    def smooth_svm_kernel_calc_update(
        self, f_grad, f_hess, grad_per_sample, hess_per_sample
    ):
        hess_minus_i = f_hess - hess_per_sample
        grad_minus_i = f_grad - grad_per_sample

        # bounds check
        if np.linalg.norm(f_hess) > 1e5:
            f_hess = np.zeros(self.p)
            hess_minus_i = -hess_per_sample

        # if we adjust the factor, we also need to add regularisation back
        hess_minus_i = self.lbd_ * self.gram_.T + self.factor * hess_minus_i
        grad_minus_i = self.lbd_ * self.gram_ @ self.u_ + self.factor * grad_minus_i

        return (grad_minus_i, hess_minus_i)

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
        warm_start=0,
        cv=False,
        approx_cv=False,
    ):
        self.gram_ = self.kernel_(X, X, **self.kernel_args_)
        self.X = X
        self.y = y

        p_nabla = lambda w, X, y: self.nabla_fgd_no_reg_no_factor_(
            w, X, y, self.sigma_, self.lbd_
        )
        p_hess = lambda w, X, y: self.hess_fgd_no_reg_no_factor_(
            w, X, y, self.sigma_, self.lbd_
        )

        p_nabla_single = lambda w, X, y: self.nabla_fgd_single_no_factor_(
            w, X, y, self.sigma_, self.lbd_
        )
        p_hess_single = lambda w, X, y: self.hess_fgd_single_no_factor_(
            w, X, y, self.sigma_, self.lbd_
        )

        self.approx_cv_obj = IACV(
            self.n,
            self.n,
            p_nabla,
            p_hess,
            p_nabla_single,
            p_hess_single,
            eta,
            calc_update=self.smooth_svm_kernel_calc_update,
        )

        p_nabla_full = lambda w, X, y: self.nabla_fgd_(w, X, y, self.sigma_, self.lbd_)
        self.true_cv_obj = TrueCV(self.n, self.n, p_nabla_full, eta)

        for t in range(n_iter):
            if approx_cv == True:
                self.approx_cv_obj.step_gd(self.u_, self.gram_, y, kernel=False)
            # grad_minus_i = np.zeros((self.n, self.n))
            # hess_minus_i = np.zeros((self.n, self.n, self.n))

            # for i in range(self.n):
            #    y_temp = y.copy()
            #    y_temp[i] = 0
            #    grad_minus_i[i] = self.nabla_fgd_(
            #        self.u_, self.gram_, y_temp, self.sigma_, self.lbd_
            #    )

            #    hess_minus_i[i] = self.hess_fgd_(
            #        self.u_, self.gram_, y_temp, self.sigma_, self.lbd_
            #    )

            # self.approx_cv_obj.iterates = (
            #    self.approx_cv_obj.iterates
            #    - eta * grad_minus_i
            #    - eta
            #    * self.approx_cv_obj.vmap_matmul(
            #        hess_minus_i, (self.approx_cv_obj.iterates - self.u_)
            #    )
            # )

            if cv == True:
                self.true_cv_obj.step_gd_kernel(self.gram_, y)

            f_grad = self.nabla_fgd_(self.u_, self.gram_, y, self.sigma_, self.lbd_)

            self.u_ = self.u_ - eta * f_grad
            # print(
            #    f"iter {t} | f_grad {np.linalg.norm(f_grad):.4f} | objective {self.SSVM_objective(self.u_, self.gram_, y, self.sigma_, self.lbd_):.4f} ",
            #    end="",
            # )

            # print(f"| accuracy: {accuracy_score(y, self.predict(X))} ", end="")

            # print(
            #    f"| IACV: {np.mean(np.linalg.norm(self.approx_cv_obj.iterates - self.true_cv_obj.iterates, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.u_ - self.true_cv_obj.iterates, 2, axis=1)):.8f}"
            # )

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

        self.factor = 1 / (self.n - 1)
        self.fit_gd_(
            X,
            y,
            eta=eta,
            n_iter=n_iter,
            **kwargs,
        )

    def predict(self, X: np.ndarray):
        n_gram = self.kernel_(self.X, X, **self.kernel_args_)
        return np.sign(self.u_ @ n_gram)
