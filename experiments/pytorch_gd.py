#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev
from sampler import sample_from_logreg

device = torch.device("cpu")


def l(X, y, theta):
    return -y * (X @ theta) + torch.log(1 + torch.exp(X @ theta))


def pi(theta):
    return torch.norm(theta, 2)


def F(X, y, theta, lbd):
    return torch.sum(l(X, y, theta)) + lbd * pi(theta)


def run_sim(n, p, n_iter=250):
    X, _, y = sample_from_logreg(p=p, n=n)

    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)

    lbd_v = 1e-6 * n

    theta = torch.zeros(p, device=device, requires_grad=True)  # grad for gd
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = [torch.zeros(p, device=device)] * n
    theta_true = [
        torch.zeros(p, requires_grad=True, device=device)
    ] * n  # grad for first-order gd
    theta_ns = [torch.zeros(p, device=device)] * n
    theta_ij = [torch.zeros(p, device=device)] * n

    def F_z(theta, lbd):
        return F(X, y, theta, lbd)

    err_approx = {
        "IACV": np.zeros(n_iter),
        "NS": np.zeros(n_iter),
        "IJ": np.zeros(n_iter),
        "hat": np.zeros(n_iter),
    }

    for t in range(0, n_iter):
        # TODO use vmap for per-sample gradient (https://pytorch.org/functorch/nightly/notebooks/per_sample_grads.html)
        f_grad = torch.autograd.grad(
            F(X, y, theta, lbd_v), theta, create_graph=False, retain_graph=False
        )[0]

        f_hess = torch.nan_to_num(
            jacrev(jacrev(F_z, argnums=0), argnums=0)(theta, lbd_v)
        )

        for i in range(0, n):
            grad_Z_i = torch.autograd.grad(
                F(X[i], y[i], theta, lbd_v),
                theta,
                create_graph=False,
                retain_graph=False,
            )[0]

            hess_Z_i = torch.nan_to_num(
                jacrev(jacrev(F, argnums=2), argnums=2)(X[i], y[i], theta, lbd_v)
            )

            hess_minus_i = f_hess - hess_Z_i
            grad_minus_i = f_grad - grad_Z_i

            theta_cv[i] = (
                theta_cv[i]
                - alpha_t * grad_minus_i
                - alpha_t * hess_minus_i @ (theta_cv[i] - theta)
            )

            theta_true[i] = (
                theta_true[i]
                - alpha
                * torch.autograd.grad(
                    F(
                        X[torch.arange(0, X.shape[0]) != i, ...],
                        y[torch.arange(0, y.shape[0]) != i, ...],
                        theta_true[i],
                        lbd_v,
                    ),
                    theta_true[i],
                    create_graph=False,
                    retain_graph=False,
                )[0]
            )

            theta_ns[i] = theta + torch.linalg.pinv(hess_minus_i) @ grad_Z_i
            theta_ij[i] = theta + torch.linalg.pinv(f_hess) @ grad_Z_i

        # actually update theta
        theta = theta - alpha * f_grad

        err_approx["IACV"][t] = torch.mean(
            torch.norm(torch.stack(theta_cv) - torch.stack(theta_true), 2, dim=1)
        )

        err_approx["NS"][t] = torch.mean(
            torch.norm(torch.stack(theta_ns) - torch.stack(theta_true), 2, dim=1)
        )

        err_approx["IJ"][t] = torch.mean(
            torch.norm(torch.stack(theta_ij) - torch.stack(theta_true), 2, dim=1)
        )

        err_approx["hat"][t] = torch.mean(
            torch.norm(theta - torch.stack(theta_true), 2, dim=1)
        )

    return err_approx
