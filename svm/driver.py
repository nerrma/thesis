#!/usr/bin/env python3
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import polynomial_kernel

from sampler import sample_from_logreg

import numpy as np
import matplotlib.pyplot as plt

from cv_svm import SVM_smooth
from kernel_svm import SVM_smooth_kernel

X, y = load_breast_cancer(return_X_y=True)
# X, y = make_classification(n_samples=100, n_features=100, random_state=10)
# X, theta_star, y = sample_from_logreg(n=250, p=20)
n = X.shape[0]
p = X.shape[1]
y[np.where(y == 0)] = -1

X = StandardScaler().fit_transform(X)

clf = SVM_smooth_kernel(sigma=1e-2, lbd=1, kernel=RBF(1))
clf.fit(
    X,
    y,
    eta=0.5 / n,
    n_iter=500,
    cv=False,
    approx_cv=True,
    log_iacv=True,
    save_err_approx=True,
    save_err_cv=True,
)

subset = np.arange(0, X.shape[0] // 2)
y_pred = clf.predict(X[subset])
print(accuracy_score(y[subset], y_pred))

# lemma_cv = clf.true_cv_obj.iterates[0]
# approx_cv = clf.approx_cv_obj.iterates[0]
#
# clf = SVM_smooth_kernel(sigma=1e-1, lbd=1, kernel=RBF(1.25))
# X_temp = np.delete(X, (0), axis=0)
# y_temp = np.delete(y, (0), axis=0)
# clf.fit(X_temp, y_temp, eta=0.5 / n, n_iter=100)
#
# print(clf.u_)
# print(lemma_cv)
# print(np.linalg.norm(lemma_cv[1:] - clf.u_, axis=0).mean())
# print(np.linalg.norm(approx_cv[1:] - clf.u_, axis=0).mean())

# plt.hist(
#    np.linalg.norm(clf.approx_cv_obj.iterates - clf.true_cv_obj.iterates, axis=0),
#    label="IACV",
#    alpha=0.3,
#    bins=60,
# )
# plt.show()
# plt.hist(
#    np.linalg.norm(clf.u_ - clf.true_cv_obj.iterates, axis=0),
#    label="true",
#    alpha=0.3,
#    bins=60,
# )
# plt.show()

# clf = SVM_smooth(sigma=1e-1, lbd=1)
# clf.fit(
#    X,
#    y,
#    thresh=1e-6,
#    n_iter=1000,
#    eta=0.85 / n,
#    approx_cv=True,
#    cv=True,
#    log_iacv=True,
#    log_iter=False,
#    log_cond_number=False,
#    log_eig_vals=False,
#    log_accuracy=True,
#    warm_start=0,
#    save_eig_vals=False,
#    save_cond_nums=True,
#    save_err_cv=True,
#    save_err_approx=True,
#    use_jax_grad=False,
#    adjust_factor=True
#    # init_w=np.ones(p),
# )

y_pred = clf.predict(X)
print(f"accuracy {accuracy_score(y, y_pred)}")

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
