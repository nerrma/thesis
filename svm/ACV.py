#!/usr/bin/env python3
#!/usr/bin/env python3
import numpy as np
from jax import vmap, jit
import jax.numpy as jnp
from jax.experimental import sparse


class ACV_Obj:
    def __init__(
        self,
        n,
        p,
        nabla_function,
        hess_function,
        nabla_single_function,
        hess_single_function,
        eta,
        calc_update=None,
    ):
        self.n = n
        self.p = p

        self.nabla_function = nabla_function
        self.hess_function = hess_function

        self.grad_Z_f = jit(vmap(nabla_single_function, in_axes=(None, 0, 0)))
        self.hess_Z_f = jit(vmap(hess_single_function, in_axes=(None, 0, 0)))

        self.grad_Z_f_kernel = jit(vmap(nabla_single_function, in_axes=(None, None, 0)))
        self.hess_Z_f_kernel = jit(vmap(hess_single_function, in_axes=(None, None, 0)))

        self.iterates = np.zeros((n, p))
        self.alpha_t = eta
        self.vmap_matmul = jit(vmap(jnp.matmul, in_axes=(0, 0)))
        self.one_axis_matmul = jit(vmap(jnp.matmul, in_axes=(None, 0)))

        self.cond_nums = []

        self.calc_update = self.calc_update_
        if calc_update is not None:
            self.calc_update = calc_update

    def calc_update_(self, f_grad, f_hess, grad_per_sample, hess_per_sample):
        # per sample gradient and hessian difference
        hess_minus_i = f_hess - hess_per_sample
        grad_minus_i = f_grad - grad_per_sample

        # bounds check
        if np.linalg.norm(f_hess) > 1e5:
            hess_minus_i = -hess_per_sample

        return (grad_minus_i, hess_minus_i)

    def step_gd(self, theta, X, y, kernel=False, save_cond_num=False, **kwargs):
        NotImplementedError()

    def step_sgd(self, theta, X, y, kernel=False, save_cond_num=False, **kwargs):
        NotImplementedError()
