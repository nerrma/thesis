#!/usr/bin/env python3

import numpy as np
import torch
import matplotlib.pyplot as plt
from sampler import sample_from_logreg
import gc


device = torch.device("cuda")


def l(X, y, theta):
    return -y * (X @ theta) + torch.log(1 + torch.exp(X @ theta))


def pi(theta):
    return torch.norm(theta, 2)


def F(X, y, theta, lbd):
    return torch.sum(l(X, y, theta)) + lbd * pi(theta)


def hess_F(X, y, theta, lbd, dot=False):
    expy = torch.exp(X @ theta) / torch.square(1 + torch.exp(X @ theta))

    if dot:
        return (X.T * expy).view(-1, 1) * X + lbd * torch.eye(
            theta.shape[0], device=device
        )
    else:
        return X.T * expy @ X + lbd * torch.eye(theta.shape[0], device=device)


def nabla_F(X, y, theta, lbd=1e-6, dot=False):
    if dot:
        return (
            -(X * y)
            + X * ((torch.exp(X @ theta)) / (1 + torch.exp(X @ theta)))
            + lbd * theta
        )
    else:
        return (
            -(X.T @ y)
            + X.T @ ((torch.exp(X @ theta)) / (1 + torch.exp(X @ theta)))
            + lbd * theta
        )


def run_sim(n, p, n_iter=250):
    X, _, y = sample_from_logreg(p=p, n=n)

    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).to(device)

    lbd_v = 1e-6 * n

    # theta = torch.zeros(p, requires_grad=True, device=device)  # grad for gd
    theta = torch.zeros(p, device=device)  # grad for gd
    alpha = 0.5 / n
    alpha_t = alpha

    theta_cv = [torch.zeros(p, device=device)] * n
    theta_true = [
        torch.zeros(p, requires_grad=False, device=device)
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
        # loss = F(X, y, theta, lbd_v)
        # print(f"loss : {loss}")
        # theta.retain_grad()
        # loss.backward(retain_graph=True)

        # f_grad = theta.grad
        # f_grad = torch.autograd.grad(F(X, y, theta, lbd_v), theta)[0]
        # f_hess = torch.func.hessian(F_z)(theta, lbd_v)
        with torch.no_grad():
            f_grad = nabla_F(X, y, theta, lbd_v)
            f_hess = hess_F(X, y, theta, lbd_v)

        for i in range(0, n):
            # grad_Z_i = torch.autograd.grad(F(X[i], y[i], theta, lbd_v), theta)[0]
            # hess_Z_i = torch.func.hessian(F)(X[i], y[i], theta, lbd_v)

            with torch.no_grad():
                grad_Z_i = nabla_F(X[i], y[i], theta, lbd_v, dot=True)
                hess_Z_i = hess_F(X[i], y[i], theta, lbd_v, dot=True)

            hess_minus_i = f_hess - hess_Z_i
            grad_minus_i = f_grad - grad_Z_i

            theta_cv[i] = (
                theta_cv[i]
                - alpha_t * grad_minus_i
                - alpha_t * hess_minus_i @ (theta_cv[i] - theta)
            )

            # theta_true[i] = (
            #    theta_true[i]
            #    - alpha
            #    * torch.autograd.grad(
            #        F(
            #            X[torch.arange(0, X.shape[0]) != i, ...],
            #            y[torch.arange(0, y.shape[0]) != i, ...],
            #            theta_true[i],
            #            lbd_v,
            #        ),
            #        theta_true[i],
            #    )[0]
            # )

            theta_true[i] = theta_true[i] - alpha * nabla_F(
                X[torch.arange(0, X.shape[0]) != i, ...],
                y[torch.arange(0, y.shape[0]) != i, ...],
                theta_true[i],
                lbd_v,
            )

            theta_ns[i] = theta + torch.inverse(hess_minus_i) @ grad_Z_i
            theta_ij[i] = theta + torch.inverse(f_hess) @ grad_Z_i

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

        if err_approx["IACV"][t] > 10:
            print("error is incorrect, exploding theta")
            print("1")
            print(theta_cv[1])
            print(theta_true[1])
            print("n-1")
            print(theta_cv[n - 1])
            print(theta_true[n - 1])
            exit(-1)

    return err_approx


# err = run_sim(250, 20, n_iter=250)
# print(f"IACV {np.mean(err['IACV'])}")
# print(f"NS {np.mean(err['NS'])}")
# print(f"IJ {np.mean(err['NS'])}")
# print(f"hat {np.mean(err['hat'])}")
