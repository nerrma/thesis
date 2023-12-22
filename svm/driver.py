#!/usr/bin/env python3
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from cv_svm import SVM_smooth

X, y = load_breast_cancer(return_X_y=True)
n = X.shape[0]
p = X.shape[1]
y[np.where(y == 0)] = -1

X = StandardScaler().fit_transform(X)

clf = SVM_smooth(sigma=2e-6, lbd=1e-5)
clf.fit(
    X,
    y,
    thresh=1e-4,
    n_iter=1000,
    eta=1e-1,
    approx_cv=True,
    cv=True,
    log_iacv=True,
    log_iter=True,
)
print(
    f"grad {np.linalg.norm(clf.nabla_fgd_(X, y, clf.weights_, clf.sigma_, clf.lbd_))}"
)
print(
    f"IACV: {np.mean(np.linalg.norm(clf.loo_iacv_ - clf.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(clf.weights_ - clf.loo_true_, 2, axis=1))}"
)
coef = clf.weights_ / np.linalg.norm(clf.weights_)

y_pred = clf.predict(X)
print(accuracy_score(y, y_pred))
