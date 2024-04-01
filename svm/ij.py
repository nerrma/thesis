#!/usr/bin/env python3
#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp
from ACV import ACV_Obj


class IJ(ACV_Obj):
    def step_gd(self, theta, X, y, save_cond_num=False, **kwargs):
        f_grad = self.nabla_function(theta, X, y)
        f_hess = self.hess_function(theta, X, y)
        f_hess_inv = jnp.linalg.inv(f_hess)

        grad_per_sample = self.grad_Z_f(theta, X, y)
        hess_per_sample = self.hess_Z_f(theta, X, y)

        _, hess_minus_i = self.calc_update(
            f_grad, f_hess, grad_per_sample, hess_per_sample, **kwargs
        )

        if save_cond_num:
            self.cond_nums.append(np.linalg.cond(hess_minus_i))

        self.iterates = theta + self.one_axis_matmul(f_hess_inv, grad_per_sample)
