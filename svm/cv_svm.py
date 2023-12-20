#!/usr/bin/env python3
import numpy as np


# An SVM class which learns using a Smooth Hinge loss
class SVM_smooth:
    def __init__(self, sigma: np.float64, lbd: np.float64):
        self.sigma_ = sigma
        self.lbd_ = lbd

        self.n = 1
        self.p = 1

        self.weights_ = np.zeros(0)

    def Phi_m(self, v):
        return (1 + v / np.sqrt(1 + v**2)) / 2

    def Phi_m_prime(self, v):
        return 1 / 2 * 1 / (np.sqrt(v**2 + 1) ** 3)

    def phi_m(self, v):
        return 1 / (2 * np.sqrt(1 + v**2))

    def nabla_fgd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * y @ X * self.Phi_m((1 - y @ X * w) / sigma)

    def nabla_fgd_single_(
        self,
        X: np.ndarray,
        y: np.float64,
        w: np.ndarray,
        sigma: np.float64,
        lbd: np.float64,
    ):
        return lbd * w - 1 / self.n * y * X * self.Phi_m((1 - y * X * w) / sigma)

    def hess_fgd_(self, X, y, w, sigma, lbd):
        d = self.Phi_m_prime((1 - y @ X * w) / sigma) / sigma
        D = np.eye(self.p)
        for i in range(self.p):
            D[i, i] = d[i]

        hess = lbd * np.eye(self.p) + 1 / self.n * (X @ D).T @ X
        return hess

    def hess_fgd_single_(self, X, y, w, sigma, lbd):
        d = self.Phi_m_prime((1 - y * X * w) / sigma) / sigma
        D = np.eye(self.p)
        for i in range(self.p):
            D[i, i] = d[i]

        hess = lbd * np.eye(self.p) + 1 / self.n * (X @ D).T * X
        return hess

    def fit_gd_(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta=1e-4,
        cv=True,
        approx_cv=True,
        n_iter=1000,
        thresh=1e-8,
        log_iacv=False,
        log_iter=False,
    ):
        alpha_t = min(eta, 1 / self.n)
        for t in range(n_iter):
            f_grad = self.nabla_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)
            f_hess = self.hess_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)

            if np.linalg.norm(f_grad) < thresh:
                print(f"stopping early at iteration {t}")
                break

            if approx_cv == True:
                # vectorised per sample hessian
                hess_per_sample = np.vectorize(
                    self.hess_fgd_single_,
                    excluded={"w", "sigma", "lbd"},
                    signature="(p),(0)->(p, p)",
                )(
                    X,
                    y.reshape(-1, 1),
                    w=self.weights_,
                    sigma=self.sigma_,
                    lbd=self.lbd_,
                )

                # vectorised per sample gradient
                grad_per_sample = np.vectorize(
                    self.nabla_fgd_single_,
                    excluded={"w", "sigma", "lbd"},
                    signature="(p),(0)->(p)",
                )(
                    X,
                    y.reshape(-1, 1),
                    w=self.weights_,
                    sigma=self.sigma_,
                    lbd=self.lbd_,
                )

                # per sample gradient and hessian difference
                hess_minus_i = f_hess - hess_per_sample
                grad_minus_i = f_grad - grad_per_sample

                self.loo_iacv_ = (
                    self.loo_iacv_
                    - alpha_t * grad_minus_i
                    - alpha_t
                    * np.vectorize(np.matmul, signature="(p, p),(p)->(p)")(
                        hess_minus_i, (self.loo_iacv_ - self.weights_)
                    )
                )

                # self.loo_iacv_ /= np.asarray(
                #    [np.linalg.norm(self.loo_iacv_, axis=0)] * self.n
                # )

                # alpha_t *= np.exp(-alpha_t)

            if cv == True or approx_cv == True:
                for i in range(self.n):
                    X_temp = np.delete(X, (i), axis=0)
                    y_temp = np.delete(y, (i), axis=0)
                    self.loo_true_[i] = self.loo_true_[i] - eta * self.nabla_fgd_(
                        X_temp, y_temp, self.weights_, self.sigma_, self.lbd_
                    )

                # self.loo_true_ /= np.asarray(
                #    [np.linalg.norm(self.loo_true_, axis=0)] * self.n
                # )

            if log_iter == True:
                print(f"iter {t} | grad {np.linalg.norm(f_grad)}")

            if log_iacv == True:
                print(
                    f"IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1))}"
                )

            self.weights_ = self.weights_ - eta * f_grad

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
