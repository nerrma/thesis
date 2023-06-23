#!/usr/bin/env python3

import numpy as np


def sample_from_logreg(p=20, n=500, seed=984):
    # if seed != 0:
    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(n, p))  # sample (n, p) from standard normal

    theta_star = np.zeros(p).reshape(-1, 1)
    non_zero = 5
    for i in np.random.choice(p, non_zero):
        theta_star[i] = np.random.normal(0, 1)

    probs = np.exp(X @ theta_star) / (np.exp(X @ theta_star) + 1)
    y = np.zeros(n)

    for i in range(0, n):
        y[i] = np.random.binomial(1, probs[i])

    return (X, theta_star, y)
