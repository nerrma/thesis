#!/usr/bin/env python3
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

from cv_svm import SVM_smooth

X, y = load_breast_cancer(return_X_y=True)
# X, y = make_classification(n_samples=1200, n_features=4, random_state=10)
n = X.shape[0]
p = X.shape[1]
y[np.where(y == 0)] = -1

X = StandardScaler().fit_transform(X)

clf = SVM_smooth(sigma=1e-1, lbd=0)
clf.fit(
    X,
    y,
    thresh=1e-3,
    n_iter=2000,
    # eta=0.5 / n,
    eta=0.8,
    approx_cv=True,
    cv=True,
    log_iacv=True,
    log_iter=True,
    log_cond_number=False,
    log_eig_vals=False,
    warm_start=1750,
    save_eig_vals=False,
    save_err_cv=True,
    save_err_approx=True,
    use_jax_grad=False,
)

y_pred = clf.predict(X)
print(
    f"accuracy {accuracy_score(y, y_pred)} | grad {np.linalg.norm(clf.nabla_fgd_(clf.weights_, X, y, clf.sigma_, clf.lbd_))}"
)
print(
    f"IACV: {np.mean(np.linalg.norm(clf.loo_iacv_ - clf.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(clf.weights_ - clf.loo_true_, 2, axis=1))}"
)
print(
    f"IACV variance: {np.var(clf.loo_iacv_)} | true variance: {np.var(clf.loo_true_)}"
)
print(f"IACV variance: {np.sort(clf.loo_iacv_)}")
print(
    f"minimum eig value (lambda_0) {np.min(clf.eig_vals_)} | maximum eig value (lambda_1) {np.max(clf.eig_vals_)}"
)

# lbds = np.linspace(0, 1, 5)
# iterates = []
# for l in lbds:
#    clf = SVM_smooth(sigma=5e-2, lbd=l)
#    clf.fit(
#        X,
#        y,
#        thresh=1e-3,
#        n_iter=2500,
#        eta=0.75 / n,
#        approx_cv=True,
#        cv=True,
#        log_iacv=False,
#        log_iter=False,
#        use_jax_grad=False,
#    )
#    print(
#        f"grad {np.linalg.norm(clf.nabla_fgd_(clf.weights_, X, y, clf.sigma_, clf.lbd_))}"
#    )
#    print(
#        f"IACV: {np.mean(np.linalg.norm(clf.loo_iacv_ - clf.loo_true_, 2, axis=1))} | baseline: {np.mean(np.linalg.norm(clf.weights_ - clf.loo_true_, 2, axis=1))}"
#    )
#    print(
#        f"IACV variance: {np.var(clf.loo_iacv_)} | true variance: {np.var(clf.loo_true_)}"
#    )
#    iterates.append(clf.loo_iacv_)
#    coef = clf.weights_ / np.linalg.norm(clf.weights_)
#
#    y_pred = clf.predict(X)
#    print(accuracy_score(y, y_pred))
#
#
# for i, it in enumerate(iterates):
#    print(f"mean of {i} {np.mean(it, axis=0)}")
