#!/usr/bin/env python3
import numpy as np
import time
from jax import vmap, jacrev, jacfwd, grad, jit
from scipy import linalg as sc_linalg
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
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
        self.cond_num_bound_ = []
        self.eig_vals_ = []
        self.err_approx_ = {"IACV": [], "baseline": []}
        self.err_cv_ = {"IACV": [], "baseline": []}
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
        d = self.Phi_m_prime((1 - y * (X @ w)) / sigma) / sigma
        hess = X.T * d @ X
        return hess

    def hess_fgd_single_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X @ w)) / sigma) / sigma
        return np.zeros((self.p, self.p)) + 1 / self.n * d * X * X

    def hess_fgd_single_no_factor_(self, w, X, y, sigma, lbd):
        d = self.Phi_m_prime_jax((1 - y * (X @ w)) / sigma) / sigma
        return np.zeros((self.p, self.p)) + d * X * X

    def loss(self, w, X, y, sigma):
        return jnp.mean(self.Psi_m(y * (X @ w), sigma))

    def SSVM_objective(self, w, X, y, sigma, lbd):
        return 1 / 2 * jnp.linalg.norm(w) * lbd + self.loss(w, X, y, sigma)

    def eval_cond_num_bound(self, X, hessian_LOO):
        n = X.shape[0]
        exclude = min(n, 50)

        vals = np.empty(exclude)
        for i, idx in enumerate(np.random.choice(np.arange(n), size=exclude)):
            d_i = np.linalg.norm(hessian_LOO[i])
            X_tilde = np.delete(X, (idx), axis=0)
            C = np.linalg.norm(X_tilde.T @ X_tilde) / (n - 1)
            # smooth this by 1/n-1?
            vals[i] = (self.lbd_ + C * d_i) / (self.lbd_)

        return np.max(vals)

        # n = X.shape[0]
        ## print(np.linalg.norm(np.linalg.norm(hessian_LOO, axis=1), axis=1).shape)
        ## print(np.linalg.norm(hessian_LOO[0]).shape)
        # i = np.argmax(np.linalg.norm(np.linalg.norm(hessian_LOO, axis=1), axis=1))
        ## print(i)
        # d_i = np.max(hessian_LOO[i])

        # X_tilde = np.delete(X, (i), axis=0)
        # assert X_tilde.shape[0] == n - 1
        # C = np.linalg.norm(X_tilde.T @ X_tilde) / (n - 1)

        # return (self.lbd_ + C * d_i) / (self.lbd_)

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
        log_eig_vals=False,
        log_accuracy=False,
        save_grads=False,
        save_hess=False,
        save_cond_nums=False,
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
        alpha_t = eta
        if adjust_factor and factor is None:
            factor = 1 / (self.n - 1)

        self.err_approx_ = {"IACV": np.empty(n_iter), "baseline": np.empty(n_iter)}
        self.err_cv_ = {"IACV": np.empty(n_iter), "baseline": np.empty(n_iter)}

        grad_Z_f = jit(vmap(self.nabla_fgd_single_, in_axes=(None, 0, 0, None, None)))
        hess_Z_f = jit(vmap(self.hess_fgd_single_, in_axes=(None, 0, 0, None, None)))

        # set per-sample gradients
        if adjust_factor:
            grad_Z_f = jit(
                vmap(self.nabla_fgd_single_no_factor_, in_axes=(None, 0, 0, None, None))
            )
            hess_Z_f = jit(
                vmap(self.hess_fgd_single_no_factor_, in_axes=(None, 0, 0, None, None))
            )

        # define gradient functions (can be variable based on arguments)
        nabla_function = self.nabla_fgd_
        hess_function = self.hess_fgd_

        # if we adjust to use a factor of n-1
        if adjust_factor:
            nabla_function = self.nabla_fgd_no_reg_no_factor_
            hess_function = self.hess_fgd_no_reg_no_factor_

        # define jax grad variables
        if use_jax_grad:
            nabla_function = jit(grad(self.SSVM_objective))
            hess_function = jit(jacfwd(jacrev(self.SSVM_objective)))

        if use_jax_grad:
            grad_Z_f = jit(vmap(nabla_function, in_axes=(None, 0, 0, None, None)))
            hess_Z_f = jit(vmap(hess_function, in_axes=(None, 0, 0, None, None)))

        vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))
        loss_vmap = jit(vmap(self.loss, in_axes=(0, 0, 0, None)))
        loss_vmap_fixed_w = jit(vmap(self.loss, in_axes=(None, 0, 0, None)))

        for t in range(n_iter):
            f_grad = nabla_function(self.weights_, X, y, self.sigma_, self.lbd_)

            if np.linalg.norm(f_grad) < thresh:
                print(f"stopping early at iteration {t}")
                break

            if approx_cv == True:
                start = time.time()
                f_hess = hess_function(self.weights_, X, y, self.sigma_, self.lbd_)

                # vectorised per sample hessian
                grad_per_sample = grad_Z_f(self.weights_, X, y, self.sigma_, self.lbd_)
                hess_per_sample = hess_Z_f(self.weights_, X, y, self.sigma_, self.lbd_)

                # per sample gradient and hessian difference
                hess_minus_i = f_hess - hess_per_sample
                grad_minus_i = f_grad - grad_per_sample

                # if we adjust the factor, we also need to add regularisation back
                if adjust_factor:
                    hess_minus_i = self.lbd_ * np.eye(self.p) + factor * hess_minus_i
                    grad_minus_i = self.lbd_ * self.weights_ + factor * grad_minus_i

                if log_cond_number or save_cond_nums:
                    cond_num = np.linalg.cond(hess_minus_i)
                    if save_cond_nums:
                        self.cond_nums_.append(cond_num)
                        self.cond_num_bound_.append(
                            self.eval_cond_num_bound(X, hess_minus_i)
                        )

                    if log_cond_number:
                        print(
                            f"hessian condition number {np.mean(np.linalg.cond(f_hess))}"
                        )
                        print(
                            f"mean hessian condition number {np.mean(cond_num)} | min hessian condition number {np.min(cond_num, axis=0)} | max hessian condition number {np.max(cond_num, axis=0)}"
                        )
                if log_eig_vals or save_eig_vals:
                    eig_vals = np.linalg.eigvals(hess_minus_i)
                    self.eig_vals_.append(eig_vals)

                    if log_eig_vals:
                        print(
                            f"min eig value {np.min(eig_vals)} | max eig value {np.max(eig_vals)}"
                        )

                if t >= warm_start:
                    if t == warm_start and warm_start != 0:
                        new_lr = eta
                        print(f"changing learning rate from {eta} to {new_lr}")
                        eta = new_lr
                        self.loo_iacv_ = np.asarray([self.weights_] * self.n)

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
                        self.loo_true_[i], X_temp, y_temp, self.sigma_, self.lbd_
                    )
                end = time.time()
                if normalise:
                    self.loo_true_ = normalize(self.loo_true_, axis=1)

            f_grad_neutral = self.nabla_fgd_(
                self.weights_, X, y, self.sigma_, self.lbd_
            )
            if log_iter == True:
                print(
                    f"iter {t} | grad {np.linalg.norm(f_grad_neutral):.5f} | objective {self.SSVM_objective(self.weights_, X, y, self.sigma_, self.lbd_):.5f} ",
                    end="",
                )

            if log_accuracy == True:
                print(f"| accuracy: {accuracy_score(y, self.predict(X))} ", end="")

            if log_iacv == True:
                print(
                    f"| IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1)):.8f} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1)):.8f}"
                )
            elif log_iter or log_accuracy:
                print("\n", end="")

            if save_err_approx:
                self.err_approx_["IACV"][t] = np.mean(
                    np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1)
                )

                self.err_approx_["baseline"][t] = np.mean(
                    np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1)
                )

            if save_err_cv:
                self.err_cv_["IACV"][t] = np.abs(
                    loss_vmap(self.loo_iacv_, X, y, self.sigma_)
                    - loss_vmap(self.loo_true_, X, y, self.sigma_)
                ).mean()

                self.err_cv_["baseline"][t] = np.abs(
                    loss_vmap_fixed_w(self.weights_, X, y, self.sigma_)
                    - loss_vmap(self.loo_true_, X, y, self.sigma_)
                ).mean()

            # update weights
            self.weights_ = self.weights_ - eta * f_grad_neutral

            if save_grads == True:
                self.grads_.append(f_grad_neutral)

            if save_hess == True:
                self.hess_.append(f_hess)

            # normalise?
            if normalise:
                self.weights_ = normalize(
                    self.weights_.reshape(-1, 1), axis=0
                ).flatten()

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
        if init_w is not None:
            self.loo_true_ = np.asarray([init_w] * self.n)
            self.loo_iacv_ = np.asarray([init_w] * self.n)

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
