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
    n_iter=10,
    cv=False,
    approx_cv=True,
    log_iter=True,
    save_err_approx=True,
    save_err_cv=True,
    sgd=False,
    batch_size=20,
)

y_pred = clf.predict(X)
print(f"Our accuracy: {accuracy_score(y, y_pred)}")

clf = SVC()
clf.fit(X, y)
y_pred = clf.predict(X)
print(f"SVC accuracy {accuracy_score(y, y_pred)}")

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
