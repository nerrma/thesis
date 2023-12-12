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
        d = self.phi_m((1 - y @ X * w) / sigma)
        D = np.eye(self.p)
        for i in range(self.p):
            D[i, i] = d[i]

        hess = lbd * np.eye(self.p) + 1 / self.n * (X @ D).T @ X
        return hess

    def hess_fgd_single_(self, X, y, w, sigma, lbd):
        d = self.phi_m((1 - y * X * w) / sigma)
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
    ):
        alpha_t = eta
        for t in range(n_iter):
            f_grad = self.nabla_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)
            f_hess = self.hess_fgd_(X, y, self.weights_, self.sigma_, self.lbd_)

            if np.linalg.norm(f_grad) < thresh:
                print(f"stopping early at iteration {t}")
                break

            if approx_cv == True:
                hess_per_sample = np.zeros((self.n, self.p, self.p))
                for i in range(self.n):
                    hess_per_sample[i] = self.hess_fgd_single_(
                        X[i], y[i], self.weights_, self.sigma_, self.lbd_
                    )

                grad_per_sample = np.zeros((self.n, self.p))
                for i in range(self.n):
                    grad_per_sample[i] = self.nabla_fgd_single_(
                        X[i], y[i], self.weights_, self.sigma_, self.lbd_
                    )

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

            if cv == True or approx_cv == True:
                for i in range(self.n):
                    X_temp = np.delete(X, (i), axis=0)
                    y_temp = np.delete(y, (i), axis=0)
                    self.loo_true_[i] = self.loo_true_[i] - eta * self.nabla_fgd_(
                        X_temp, y_temp, self.weights_, self.sigma_, self.lbd_
                    )

            self.weights_ = self.weights_ - eta * f_grad
            # print(
            #    f"{self.weights_} | IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1))}"
            # )
            print(
                f"IACV: {np.mean(np.linalg.norm(self.loo_iacv_ - self.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(self.weights_ - self.loo_true_, 2, axis=1))}"
            )
            # print(f"iter {t} | grad {np.linalg.norm(f_grad)}")

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
