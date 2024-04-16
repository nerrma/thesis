#!/usr/bin/env python3
import numpy as np
from ACV import ACV_Obj


class IACV(ACV_Obj):
    def step_gd(self, theta, X, y, kernel=False, save_cond_num=False, **kwargs):
        f_grad = self.nabla_function(theta, X, y)
        f_hess = self.hess_function(theta, X, y)

        if not kernel:
            grad_per_sample = self.grad_Z_f(theta, X, y)
            hess_per_sample = self.hess_Z_f(theta, X, y)
        else:
            grad_per_sample = self.grad_Z_f_kernel(theta, X, y)
            hess_per_sample = self.hess_Z_f_kernel(theta, X, y)

        grad_minus_i, hess_minus_i = self.calc_update(
            f_grad, f_hess, grad_per_sample, hess_per_sample, **kwargs
        )

        if save_cond_num:
            self.cond_nums.append(np.linalg.cond(hess_minus_i))

        self.iterates = (
            self.iterates
            - self.alpha_t * grad_minus_i
            - self.alpha_t * self.vmap_matmul(hess_minus_i, (self.iterates - theta))
        )

    def step_sgd(
        self, theta, X_batch, y_batch, idxs, kernel=False, save_cond_num=False, **kwargs
    ):
        f_grad = self.nabla_function(theta, X_batch, y_batch)
        f_hess = self.hess_function(theta, X_batch, y_batch)

        grad_per_sample = self.grad_Z_f(theta, X_batch, y_batch)
        hess_per_sample = self.hess_Z_f(theta, X_batch, y_batch)

        grad_minus_i, hess_minus_i = self.calc_update(
            f_grad, f_hess, grad_per_sample, hess_per_sample, **kwargs
        )

        if save_cond_num:
            self.cond_nums.append(np.linalg.cond(hess_minus_i))

        self.iterates[idxs] = (
            self.iterates[idxs]
            - self.alpha_t * grad_minus_i
            - self.alpha_t
            * self.vmap_matmul(hess_minus_i, (self.iterates[idxs] - theta))
        )
