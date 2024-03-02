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
        self.err_approx_ = {"IACV": [], "baseline": []}
        self.err_cv_ = {"IACV": [], "baseline": []}

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
        n = gram.shape[0]
        d = self.Phi_m_prime((1 - y * (gram @ u)) / sigma) / sigma
        hess = np.zeros((n, n)) + gram.T * d @ gram

        return hess

    def hess_fgd_single_(self, u, gram, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (gram @ u)) / sigma) / sigma
        hess = lbd * gram + 1 / self.n * gram.T * d @ gram

        return hess

    def hess_fgd_single_no_factor_(self, u, gram, y, sigma, lbd):
        n = gram.shape[0]
        d = self.Phi_m_prime_jax((1 - y * (gram @ u)) / sigma) / sigma
        hess = np.zeros((n, n)) + gram.T * d @ gram

        return hess

    def smooth_svm_kernel_calc_update(
        self, f_grad, f_hess, grad_per_sample, hess_per_sample, batch_idxs=[]
    ):
        hess_minus_i = f_hess - hess_per_sample
        grad_minus_i = f_grad - grad_per_sample

        # bounds check
        if np.linalg.norm(f_hess) > 1e5:
            f_hess = np.zeros(self.p)
            hess_minus_i = -hess_per_sample

        # if we are in sgd, extend the arrays to ensure we have the right shape
        if len(batch_idxs) > 0:
            hess_ext = np.zeros((self.n, self.n, self.n))
            hess_ext[np.ix_(batch_idxs, batch_idxs, batch_idxs)] = hess_minus_i
            hess_minus_i = hess_ext

            grad_ext = np.zeros((self.n, self.n))
            grad_ext[np.ix_(batch_idxs, batch_idxs)] = grad_minus_i
            grad_minus_i = grad_ext

        # if we adjust the factor, we also need to add regularisation back
        hess_minus_i = self.lbd_ * self.gram_.T + self.factor * hess_minus_i
        grad_minus_i = self.lbd_ * self.gram_ @ self.u_ + self.factor * grad_minus_i

        return (grad_minus_i, hess_minus_i)

    def loss(self, u, gram, y, sigma):
        return jnp.mean(self.Psi_m(y * (gram @ u), sigma))

    def SSVM_objective(self, u, gram, y, sigma, lbd):
        return 1 / 2 * (u.T @ gram @ u) * lbd + self.loss(u, gram, y, sigma)

    def run_fit_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-3,
        n_iter=1000,
        warm_start=0,
        cv=False,
        approx_cv=False,
        log_iacv=False,
        log_accuracy=False,
        log_iter=False,
        save_err_approx=False,
        save_err_cv=False,
        sgd=False,
        batch_size=0,
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

        self.err_approx_ = {"IACV": np.empty(n_iter), "baseline": np.empty(n_iter)}
        self.err_cv_ = {"IACV": np.empty(n_iter), "baseline": np.empty(n_iter)}
        p_nabla_full = lambda w, X, y: self.nabla_fgd_(w, X, y, self.sigma_, self.lbd_)
        self.true_cv_obj = TrueCV(self.n, self.n, p_nabla_full, eta)
        loss_vmap = jit(vmap(self.loss, in_axes=(0, 0, 0, None)))
        loss_vmap_fixed_w = jit(vmap(self.loss, in_axes=(None, 0, 0, None)))

        batch_idxs = []
        batch_u = np.empty(0)
        batch_gram = np.empty(0)
        batch_y = np.empty(0)
        for t in range(n_iter):
            if sgd:
                batch_idxs = np.random.choice(np.arange(self.n), size=batch_size)
                batch_u = self.u_[batch_idxs]
                batch_gram = self.gram_[np.ix_(batch_idxs, batch_idxs)]
                batch_y = y[batch_idxs]

            if approx_cv == True:
                if sgd:
                    self.approx_cv_obj.step_gd(
                        batch_u,
                        batch_gram,
                        batch_y,
                        full_theta=self.u_,
                        kernel=False,
                        batch_idxs=batch_idxs,
                    )
                else:
                    self.approx_cv_obj.step_gd(self.u_, self.gram_, y, kernel=False)

            if cv == True:
                if sgd:
                    self.true_cv_obj.step_gd_kernel(batch_gram, batch_y)
                else:
                    self.true_cv_obj.step_gd_kernel(self.gram_, y)

            if sgd:
                f_grad = np.zeros(self.n)
                f_grad[batch_idxs] = self.nabla_fgd_(
                    batch_u, batch_gram, batch_y, self.sigma_, self.lbd_
                )
            else:
                f_grad = self.nabla_fgd_(self.u_, self.gram_, y, self.sigma_, self.lbd_)
            self.u_ = self.u_ - eta * f_grad

            if log_iter == True:
                print(
                    f"iter {t} | grad {np.linalg.norm(f_grad):.5f} | objective {self.SSVM_objective(self.u_, self.gram_, y, self.sigma_, self.lbd_):.5f} ",
                    end="",
                )

            if log_accuracy == True:
                print(f"| accuracy: {accuracy_score(y, self.predict(X))} ", end="")

            if log_iacv == True:
                print(
                    f"| IACV: {np.mean(np.linalg.norm(self.approx_cv_obj.iterates - self.true_cv_obj.iterates, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.u_ - self.true_cv_obj.iterates, 2, axis=1)):.8f}"
                )
            elif log_iter or log_accuracy:
                print("\n", end="")

            if save_err_approx:
                self.err_approx_["IACV"][t] = np.mean(
                    np.linalg.norm(
                        self.approx_cv_obj.iterates - self.true_cv_obj.iterates,
                        2,
                        axis=1,
                    )
                )

                self.err_approx_["baseline"][t] = np.mean(
                    np.linalg.norm(self.u_ - self.true_cv_obj.iterates, 2, axis=1)
                )

            if save_err_cv:
                self.err_cv_["IACV"][t] = np.abs(
                    loss_vmap(self.approx_cv_obj.iterates, self.gram_, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, self.gram_, y, self.sigma_)
                ).mean()

                self.err_cv_["baseline"][t] = np.abs(
                    loss_vmap_fixed_w(self.u_, self.gram_, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, self.gram_, y, self.sigma_)
                ).mean()

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
        self.run_fit_(
            X,
            y,
            eta=eta,
            n_iter=n_iter,
            **kwargs,
        )

    def predict(self, X: np.ndarray):
        n_gram = self.kernel_(self.X, X, **self.kernel_args_)
        return np.sign(self.u_ @ n_gram)
