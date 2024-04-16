#!/usr/bin/env python3
import numpy as np
import time
from jax import vmap, jacrev, jacfwd, grad, jit
from scipy import linalg as sc_linalg
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import jax.numpy as jnp
from iacv import IACV
from ns import NS
from ij import IJ
from true_cv import TrueCV


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

    def nabla_fgd_no_reg_no_factor_(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return -self.Phi_m((1 - y * (X @ w)) / sigma) * y @ X

    def nabla_fgd_single_(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.float64,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return -1 / self.n * self.Phi_m_jax((1 - y * (X * w)) / sigma) * y * X

    def nabla_fgd_single_no_factor_(
        self,
        w: np.ndarray,
        X: np.ndarray,
        y: np.float64,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return -self.Phi_m_jax((1 - y * (X * w)) / sigma) * y * X

    def hess_fgd_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime((1 - y * (X @ w)) / sigma) / sigma
        hess = lbd * np.eye(self.p) + 1 / self.n * X.T * d @ X

        return hess

    def hess_fgd_no_reg_no_factor_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X @ w)) / sigma) / sigma
        hess = X.T * d @ X
        return hess

    def hess_fgd_single_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X * w)) / sigma) / sigma
        return np.zeros((self.p, self.p)) + 1 / self.n * d * X * X

    def hess_fgd_single_no_factor_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X * w)) / sigma) / sigma
        return np.zeros((self.p, self.p)) + d * X * X

    def loss(self, w, X, y, sigma):
        return jnp.mean(self.Psi_m(y * (X @ w), sigma))

    def SSVM_objective(self, w, X, y, sigma, lbd):
        return 1 / 2 * jnp.linalg.norm(w) * lbd + self.loss(w, X, y, sigma)

    def eval_cond_num_bound(self, X, hessian_LOO):
        n = X.shape[0]
        i = np.argmax(np.linalg.norm(np.linalg.norm(hessian_LOO, axis=1), axis=1))
        d_i = np.max(hessian_LOO[i])

        X_tilde = np.delete(X, (i), axis=0)
        assert X_tilde.shape[0] == n - 1
        C = np.linalg.norm(X_tilde.T @ X_tilde) / (n - 1)

        return (self.lbd_ + C * d_i) / (self.lbd_)

    def smooth_svm_calc_update(self, f_grad, f_hess, grad_per_sample, hess_per_sample):
        hess_minus_i = f_hess - hess_per_sample
        grad_minus_i = f_grad - grad_per_sample

        # bounds check
        if np.linalg.norm(f_hess) > 1e5:
            f_hess = np.zeros(self.p)
            hess_minus_i = -hess_per_sample

        # if we adjust the factor, we also need to add regularisation back
        hess_minus_i = self.lbd_ * np.eye(self.p) + self.factor * hess_minus_i
        grad_minus_i = self.lbd_ * self.weights_ + self.factor * grad_minus_i

        return (grad_minus_i, hess_minus_i)

    def fit_gd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-4,
        cv=False,
        approx_cv=False,
        approx_cv_types=["IACV"],
        n_iter=1000,
        thresh=1e-8,
        log_iacv=False,
        log_iter=False,
        sgd=False,
        batch_size=20,
        log_cond_number=False,
        log_eig_vals=False,
        log_accuracy=False,
        save_grads=False,
        save_cond_nums=False,
        save_hessian_norms=False,
        save_eig_vals=False,
        save_err_approx=False,
        save_err_cv=False,
        use_jax_grad=False,
        warm_start=0,
        normalise=False,
        adjust_factor=True,
        factor=None,
    ):
        # ensure we have a valid factor (if needed) and learning rate
        self.factor = factor
        if adjust_factor and factor is None:
            self.factor = 1 / (self.n - 1)

        self.err_approx_ = {
            "IACV": np.empty(n_iter),
            "baseline": np.empty(n_iter),
            "NS": np.empty(n_iter),
            "IJ": np.empty(n_iter),
        }
        self.err_cv_ = {
            "IACV": np.empty(n_iter),
            "baseline": np.empty(n_iter),
            "NS": np.empty(n_iter),
            "IJ": np.empty(n_iter),
        }
        loss_vmap = jit(vmap(self.loss, in_axes=(0, 0, 0, None)))
        loss_vmap_fixed_w = jit(vmap(self.loss, in_axes=(None, 0, 0, None)))

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
            self.p,
            p_nabla,
            p_hess,
            p_nabla_single,
            p_hess_single,
            eta,
            calc_update=self.smooth_svm_calc_update,
        )

        self.ns_cv_obj = NS(
            self.n,
            self.p,
            p_nabla,
            p_hess,
            p_nabla_single,
            p_hess_single,
            eta,
            calc_update=self.smooth_svm_calc_update,
        )

        self.ij_cv_obj = IJ(
            self.n,
            self.p,
            p_nabla,
            p_hess,
            p_nabla_single,
            p_hess_single,
            eta,
            calc_update=self.smooth_svm_calc_update,
        )

        p_nabla_full = lambda w, X, y: self.nabla_fgd_(w, X, y, self.sigma_, self.lbd_)
        self.true_cv_obj = TrueCV(self.n, self.p, p_nabla_full, eta)

        for t in range(n_iter):
            idxs = np.random.choice(np.arange(0, self.n), size=batch_size)
            if approx_cv == True:
                start = time.time()
                if "IACV" in approx_cv_types:
                    if sgd:
                        self.approx_cv_obj.step_sgd(
                            self.weights_,
                            X[idxs],
                            y[idxs],
                            idxs,
                            save_cond_num=save_cond_nums,
                        )
                    else:
                        self.approx_cv_obj.step_gd(
                            self.weights_, X, y, save_cond_num=save_cond_nums
                        )

                if "NS" in approx_cv_types:
                    self.ns_cv_obj.step_gd(
                        self.weights_, X, y, save_cond_num=save_cond_nums
                    )

                if "IJ" in approx_cv_types:
                    self.ij_cv_obj.step_gd(
                        self.weights_, X, y, save_cond_num=save_cond_nums
                    )
                end = time.time()

            if cv == True:
                start = time.time()
                if sgd:
                    self.true_cv_obj.step_sgd(X[idxs], y[idxs], idxs)
                else:
                    self.true_cv_obj.step_gd(X, y)
                end = time.time()

            f_grad_neutral = jnp.empty(())
            if sgd:
                f_grad_neutral = self.nabla_fgd_(
                    self.weights_, X[idxs], y[idxs], self.sigma_, self.lbd_
                )
            else:
                f_grad_neutral = self.nabla_fgd_(
                    self.weights_, X, y, self.sigma_, self.lbd_
                )

            # update weights
            self.weights_ = self.weights_ - eta * f_grad_neutral

            if np.linalg.norm(f_grad_neutral) < thresh:
                print(f"stopping early at iteration {t}")
                for k in self.err_cv_:
                    self.err_cv_[k] = self.err_cv_[k][:t]

                for k in self.err_approx_:
                    self.err_approx_[k] = self.err_approx_[k][:t]
                break

            if log_iter == True:
                print(
                    f"iter {t} | grad {np.linalg.norm(f_grad_neutral):.5f} | objective {self.SSVM_objective(self.weights_, X, y, self.sigma_, self.lbd_):.5f} ",
                    end="",
                )

            if log_accuracy == True:
                print(f"| accuracy: {accuracy_score(y, self.predict(X))} ", end="")

            if log_iacv == True:
                print(
                    f"| NS: {np.mean(np.linalg.norm(self.ns_cv_obj.iterates - self.true_cv_obj.iterates, 2, axis=1)):.8f} ",
                    end="",
                )
                print(
                    f"| IACV: {np.mean(np.linalg.norm(self.approx_cv_obj.iterates - self.true_cv_obj.iterates, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.true_cv_obj.iterates, 2, axis=1)):.8f}"
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

                self.err_approx_["NS"][t] = np.mean(
                    np.linalg.norm(
                        self.ns_cv_obj.iterates - self.true_cv_obj.iterates,
                        2,
                        axis=1,
                    )
                )

                self.err_approx_["IJ"][t] = np.mean(
                    np.linalg.norm(
                        self.ij_cv_obj.iterates - self.true_cv_obj.iterates,
                        2,
                        axis=1,
                    )
                )

                self.err_approx_["baseline"][t] = np.mean(
                    np.linalg.norm(self.weights_ - self.true_cv_obj.iterates, 2, axis=1)
                )

            if save_err_cv:
                self.err_cv_["IACV"][t] = np.abs(
                    loss_vmap(self.approx_cv_obj.iterates, X, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, X, y, self.sigma_)
                ).mean()

                self.err_cv_["NS"][t] = np.abs(
                    loss_vmap(self.ns_cv_obj.iterates, X, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, X, y, self.sigma_)
                ).mean()

                self.err_cv_["IJ"][t] = np.abs(
                    loss_vmap(self.ij_cv_obj.iterates, X, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, X, y, self.sigma_)
                ).mean()

                self.err_cv_["baseline"][t] = np.abs(
                    loss_vmap_fixed_w(self.weights_, X, y, self.sigma_)
                    - loss_vmap(self.true_cv_obj.iterates, X, y, self.sigma_)
                ).mean()

            if save_grads == True:
                self.grads_.append(f_grad_neutral)

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
