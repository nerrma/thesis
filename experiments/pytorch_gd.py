#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.func import jacrev, jacfwd, grad
from sampler import sample_from_logreg

device = torch.device("cpu")


def l(X, y, theta):
    return -y * (X @ theta) + torch.log(1 + torch.exp(X @ theta))


def pi(theta):
    return torch.norm(theta, 2)


def F_mod(theta, X, y, lbd: float):
    return torch.sum(
        -y * (X @ theta) + torch.log(1 + torch.exp(X @ theta))
    ) + lbd * torch.norm(theta, 2)


def run_sim(n, p, n_iter=250):
    X, theta_star, y = sample_from_logreg(p=p, n=n)

    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)

    lbd_v = 1e-6 * n

    theta = torch.zeros(p, device=device, requires_grad=True)  # grad for gd
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = torch.zeros((n, p), device=device)
    theta_true = [
        torch.zeros(p, requires_grad=True, device=device)
    ] * n  # grad for first-order gd
    # theta_true = torch.nested.nested_tensor(
    #    [torch.zeros(p, requires_grad=True, device=device)] * n
    # )  # grad for first-order gd
    theta_ns = torch.zeros((n, p), device=device)
    theta_ij = torch.zeros((n, p), device=device)

    err_approx = {
        "IACV": torch.zeros(n_iter),
        "NS": torch.zeros(n_iter),
        "IJ": torch.zeros(n_iter),
        "hat": torch.zeros(n_iter),
    }

    nabla_F = grad(F_mod)
    hess_F = jacrev(jacrev(F_mod))
    vmap_matmul = torch.vmap(torch.matmul, in_dims=(0, 0))

    mask = ~torch.diag(torch.ones(n, dtype=torch.bool))

    with torch.no_grad():
        for t in range(0, n_iter):
            f_grad = nabla_F(theta, X, y, lbd_v)
            f_hess = torch.nan_to_num(hess_F(theta, X, y, lbd_v))

            grad_Z = torch.vmap(nabla_F, in_dims=(None, 0, 0, None))(theta, X, y, lbd_v)
            hess_Z = torch.nan_to_num(
                torch.vmap(hess_F, in_dims=(None, 0, 0, None))(theta, X, y, lbd_v)
            )

            grad_minus = f_grad - grad_Z
            hess_minus = f_hess - hess_Z

            theta_cv = (
                theta_cv
                - alpha_t * grad_minus
                - alpha_t * vmap_matmul(hess_minus, (theta_cv - theta))
            )

            for i in range(n):
                theta_true[i] = theta_true[i] - alpha * nabla_F(
                    theta_true[i], X[mask[i, :]], y[mask[i, :]], lbd_v
                )

            theta_ns = theta + vmap_matmul(torch.linalg.pinv(hess_minus), grad_Z)
            theta_ij = theta + torch.vmap(torch.matmul, in_dims=(None, 0))(
                torch.linalg.pinv(f_hess), grad_Z
            )

            # actually update theta
            theta = theta - alpha * f_grad

            true_stack = torch.stack(theta_true)
            err_approx["IACV"][t] = torch.mean(
                torch.norm(theta_cv - true_stack, 2, dim=1)
            )
            err_approx["NS"][t] = torch.mean(
                torch.norm(theta_ns - true_stack, 2, dim=1)
            )
            err_approx["IJ"][t] = torch.mean(
                torch.norm(theta_ij - true_stack, 2, dim=1)
            )
            err_approx["hat"][t] = torch.mean(torch.norm(theta - true_stack, 2, dim=1))

    # print(torch.mean(torch.stack(theta_true), dim=1))
    print(torch.mean(theta_cv, dim=0).detach().numpy().ravel())
    print(theta_star.ravel())
    return {k: v.detach().numpy() for k, v in err_approx.items()}
