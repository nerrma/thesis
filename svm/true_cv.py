#!/usr/bin/env python3
import numpy as np


class TrueCV:
    def __init__(
        self,
        n,
        p,
        nabla_function,
        eta,
    ):
        self.n = n
        self.p = p

        self.nabla_function = nabla_function

        self.iterates = np.zeros((n, p))
        self.eta = eta

    def step_gd(self, X, y):
        for i in range(self.n):
            X_temp = np.delete(X, (i), axis=0)
            y_temp = np.delete(y, (i), axis=0)
            self.iterates[i] = self.iterates[i] - self.eta * self.nabla_function(
                self.iterates[i], X_temp, y_temp
            )

    def step_gd_kernel(self, gram, y):
        for i in range(self.n):
            # zero out gram matrix
            gram_temp = gram[i, :].copy()
            gram[i, :] = np.zeros_like(gram[i, :])
            gram[:, i] = np.zeros_like(gram[i, :])

            y_temp = y[i]
            y[i] = 0
            self.iterates[i] = self.iterates[i] - self.eta * self.nabla_function(
                self.iterates[i], gram, y
            )

            gram[i, :] = gram_temp
            gram[:, i] = gram_temp.T
            y[i] = y_temp
