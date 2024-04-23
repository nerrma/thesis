---
header-includes: |
	\usepackage{amsmath}
	 \usepackage{fancyhdr}
	 \usepackage{pgfplots}
	 \usepackage{physics}
	 \usepackage{hyperref}
	 \usepackage{subfigure}
	 \usepackage{graphicx}
	 \usepackage{xcolor}
	 \usepackage{epstopdf}
	\graphicspath{ {./figures/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
	\newcommand{\nbTh}{\nabla_{\theta}}
	\newcommand{\Rreg}{R_{\text{reg}}}
	\newcommand{\Remp}{R_{\text{emp}}}
	\def\mathdefault#1{#1}
title: "Iterative Approximate Cross Validation in High Dimensions"
author: "Sina Alsharif"
theme: "Frankfurt"
institute: "School of Computer Science and Engineering, University of New South Wales"
topic: "Thesis C"
date: "Thesis C (Term 1, 2024)"
colortheme: "orchid"
fonttheme: "professionalfonts"
toc: true
---

# Background
To standardise the notation used throughout this seminar, we define the Empirical Risk Minimisation (ERM) framework used to solve a supervised learning problem. 
\pause

### General Problem Setting
- Input space $\mathcal{X}$ and an output space $\mathcal{Y}$.
- Data is ``generated'' by a *true* distribution $P(\mathcal{X}, \mathcal{Y})$.
- Find a mapping $h : \mathcal{X} \to \mathcal{Y}$ (called a hypothesis).
- All possible combinations of input and output space are $\mathcal{D} = \{(X, Y) \in (\mathcal{X}, \mathcal{Y})\}$.

## Risk

To measure the error (or loss) we make on a data point, define a function $\ell(h; D_i)$,
\begin{align*}
	\ell(h; D) = \sum_{i=1}^n \ell(h; D_i) & \text{ where } D_i = (X_i, Y_i).
\end{align*}

\pause

### Risk
Define the **risk** of a hypothesis for the complete space of inputs and outputs as,
\begin{align*}
	R(h) = \mathbb{E}_{\mathcal{X}, \mathcal{Y}} [\ell(h; \mathcal{D})].
\end{align*}

We define **empirical risk** of a hypothesis given data $D$ as,
\begin{align*}
	R_{\text{emp}}(h; D) = \ell(h; D).
\end{align*}

## Issues with ERM

The minimiser of sample risk is not necessarily the minimiser of true risk.

We can restrict the learning algorithm's ability to fully minimise empirical risk through **regularisation**.

\pause

### Regularised Empirical Risk
To restrict the hypothesis space, we define the formulation of **regularised empirical risk** as,
\begin{align*}
	R_{\text{reg}}(h; D) = R_{\text{emp}}(h; D) + \lambda \pi(\theta).
\end{align*}
$\pi : \mathbb{R}^p \to \mathbb{R}$ is a regulariser and $\lambda \in \mathbb{R}^+$ is controls the strength of regularisation.

## Cross Validation
To avoid ``overfitting'', we need to define an estimate of true risk from empirical data.

\pause

We can do this through Cross Validation (CV).

\pause
Leave One Out Cross Validation (LOOCV) is effective ^[@CVSurvey], but costly.

# Literature Review

## Approximate CV Methods
To reduce the computational cost LOOCV, we turn to Approximate Cross Validation (ACV).

### LOOCV
The definition of leave-one-out (LOO) regularised empirical risk is,
\begin{align*}
	R_{\text{reg}}(\theta; D_{-j}) = \sum_{i=1,\,i \neq j}^n \ell(\theta; D_i) + \lambda \pi(\theta)
\end{align*}
where we leave out a point with index $j$ for this experiment.

## Methods for ACV

There are three main methods for ACV in the literature. 

- Newton Step (``NS'')
- Infinitesimal Jackknife (``IJ'')
- Iterative Approximate Cross Validation (``IACV'')

both NS and IJ are existing methods, and IACV is a new proposed method which we aim to adapt and extend. \pause 

We present a short summary of the theory behind these approximations.

## Newton Step

In summary, we take a Newton Step on an approximation of the objective function. \pause

\textcolor{blue}{Note that we assume $\ell(\mathord{\cdot})$ and $\pi(\mathord{\cdot})$ are both continuous and twice-differentiable functions, also that the Hessian is invertible.} \pause

For discussion, the standard notation we use for the NS method is,
\begin{align*}
    \tilde{\theta}^{-i}_{\text{NS}} = \mathcolor{red}{\hat{\theta}} + \left(H(\hat{\theta}; D) - \nbTh^2 \ell(\hat{\theta}; D_{i})\right)^{-1} \nbTh R_{\text{reg}}(\hat{\theta}; D_i)
\end{align*}

\textcolor{red}{The quality of the approximation depends heavily on the fact that $\hat{\theta} \approx \theta^*$.}

## Infinitesimal Jackknife

The general idea is again to perform a first-order Taylor expansion to approximate LOOCV, around the weights of a Jackknife. 

The final form derived for this case is
\begin{align*}
    \tilde{\theta}^{-i}_{\text{IJ}} = \textcolor{red}{\hat{\theta}} + (H(\hat{\theta}; D))^{-1} \nabla_\theta R_{\text{reg}}(\hat{\theta}; D_i)
\end{align*}
with the same assumptions as in NS (\textcolor{blue}{loss and regularisation are continuously twice-differentiable, $H$ is invertible} and \textcolor{red}{$\hat{\theta} \approx \theta^*$}).

## Iterative Approximate Cross Validation
Iterative Approximate Cross Validation (IACV) relaxes the assumptions of previous methods.

We solve the main learning task through updates,
\begin{align*}
    \hat{\theta}^{(k)} = \hat{\theta}^{(k-1)} - \alpha_k \nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k})
\end{align*}
for $S_k \subseteq [n]$ as a subset of indices and $\alpha_k$. \pause

The LOO update excluding a point $i$ is defined as,
\begin{align*}
    \hat{\theta}^{(k)}_{-i} = \hat{\theta}^{(k-1)}_{-i} - \alpha_k \nbTh \Rreg(\hat{\theta}^{(k-1)}_{-i}; D_{S_k \setminus i})
\end{align*}
this step is what we aim to approximate.

--- 

The cost is in evaluating $\nbTh \Rreg(\hat{\theta}^{(k-1)}_{-i}; D_{S_t \setminus i})$ for $n$ points. \pause We use a Taylor expansion of the Jacobian for $\hat{\theta}^{(k-1)}_{-i}$ centered around the estimate $\hat{\theta}^{(k-1)}$ for approximation. \pause Here,
\begin{align*}
    \nbTh \Rreg(\hat{\theta}^{(k-1)}_{-i}; D_{S_k \setminus i}) \approx \underbrace{\nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) + \nbTh^2 \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)}_{\tilde J_{-i}^{(k)}}
\end{align*}

\pause
Therefore, the IACV updates for GD and SGD become,
\begin{align*}
    \only<4>{\tilde{\theta}^{(k)}_{-i} &= \tilde{\theta}^{(k-1)}_{-i} - \alpha_k \tilde J_{-i}^{(k)} \\}
    \only<5->{\tilde{\theta}^{(k)}_{-i} &= \tilde{\theta}^{(k-1)}_{-i} - \alpha_k\left(\nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) + \nbTh^2 \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)\right)}
\end{align*}

---

We swap a Jacobian for a Jacobian *and* a Hessian in the approximation step. \pause The time complexities for the operations are as follows,

\begin{center}
	\begin{tabular}{c|c|c}
		& IACV & Exact LOOCV \\
		\hline
		GD & $n(A_p + B_p) + np^2$ & $n^2 A_p + np$ \\
		SGD & $K(A_p + B_p) + np^2$ & $nKA_p + np$ \\
		ProxGD & $n(A_p + B_p + D_p) + np^2$ & $n^2A_p + nD_p + np$ \\
	\end{tabular}
\end{center}
where $A_p$ is one evaluation of the Jacobian, $B_p$ is one evaluation of the Hessian, $D_p$ is one evaluation of the proximal operator and $K$ is the size of the subset used for SGD ^[@IACV]. \pause 

We can however parallelise calculations while keeping $\hat{\theta}^{(k-1)}$ fixed for a measurable speedup.

---

Define metrics for measuring ACV error,

$$
\text{Err}_{\text{Approx}} = \frac{1}{n} \sum_{i=1}^n \|\tilde{\theta}_{-i}^{(k)} - \hat{\theta}_{-i}^{(k)}\|_2^2
$$

\pause

$$
\text{Err}_{\text{CV}} = \frac{1}{n} \sum_{i=1}^n \left|\ell\left(\tilde{\theta}_{-i}^{(k)}; D_i\right) - \ell\left(\hat{\theta}_{-i}^{(k)}; D_i\right)\right|.
$$

\pause

The standard dataset $D = \{(X_i, Y_i)\}_{i=1}^n$ for $X_i \in \mathbb{R}^p$ and $Y_i \in \{0, 1\}$ is generated by

$$
    Y_i \sim \mathrm{Bernoulli}\left(\frac{1}{1 + \exp(-X_i^T \theta^*)}\right)
$$

for $\theta^* \in \mathbb{R}^p$ with 5 non-zero entries.

---

We recreate the experiment shown in the paper ($n = 250, p = 20$) as a sanity check of the Python implementation.

:::: columns
::: column
\begin{figure}
    \centering
	\vspace{5.75mm}
    \scalebox{0.55}{\input{figures/err_approx_250.pgf}}
    \caption{My implementation (run for less iterations).}
\end{figure}
:::
::: column
\begin{figure}
    \centering
    \includegraphics[scale=0.375]{figures/err_approx_250_iacv.png}
    \caption{Experiment in the paper.}
\end{figure}
:::
::::

The 'baseline' we measure is using $\hat{\theta}^{(k)}$ as LOOCV estimates.

---

## Gaps in the Literature

There are current gaps in the literature that we aim to address.

- \textcolor{gray}{Problems in high dimensions}
- Lack of standard (or optimised) Python implementation of IACV
- No application of ACV to non GLM settings
- No attempt at applying ACV to models without a Hessian

## Smooth Hinge Loss
The classic form of the hinge loss is not twice-differentiable, so we cannot apply ACV to any algorithm which uses it (SVM). 
\begin{align*}
	h(z) = \max\{0, 1 - z\}
\end{align*}
\pause
We therefore seek a smooth approximation which is twice-differentiable to model the same problem an apply ACV for a fast approximation of true risk. 

---

The paper [@luo2021learning], introduces a Smooth Hinge Loss of the form
\begin{align*}
    \psi_M(z; \sigma) = \Phi_M(v)(1 - z) + \phi_M(v)\sigma,
\end{align*}
where $v = (1-z)/\sigma$ and $\sigma > 0$ controls the *smoothness* of the loss.

\pause

The individual functions which make up the loss are,
\begin{align*}
    \Phi_M(v) &= \frac{1}{2} \left(1 + \frac{v}{\sqrt{1 + v^2}}\right), \\
    \phi_M(v) &= \frac{1}{2\sqrt{1 + v^2}}.
\end{align*}

---

We then use this loss to define the main objective,
\begin{align*}
    L(w; D, \sigma) = \frac{\lambda}{2} \|w\|_2^2 + \frac{1}{n} \sum_{i=1}^n \psi_M(Y_i w^T X_i; \sigma)
\end{align*}
where $\lambda$ controls the size of the margin.

---

\begin{figure}
	\resizebox{14.5cm}{!}{\input{figures/smoothhinge_sigma_comb.pgf}}
    \caption{Smooth Hinge and its derivative for varying $\sigma$. Values for the original hinge loss are shown dotted.}
    \label{fig:smoothhinge_sigma}
\end{figure}

---

\begin{figure}
    \input{figures/smoothhinge_sigma_hessian.pgf}
    \caption{Second derivative of smooth hinge loss.}
\end{figure}

---

We can solve the Smooth SVM problem with GD and achieve results similar to solving SVM using QP (`libsvm`).

\begin{figure}
    \resizebox{6cm}{!}{\input{figures/svmtest_sigma_accuracy.pgf}}
    \caption{5-Fold Accuracy of Smooth SVM solved using GD.}
\end{figure}

`libsvm` achieves a mean $97.88\%$ accuracy on the same dataset (breast cancer binary classification from scikit-learn ^[@scikit-learn]).

# Contributions
## ACV Applied to Smooth Hinge
Previous attempts at applying ACV to problems without a Hessian are lacking. The Smooth SVM problem allows for ACV through its Hessian approximation.

\pause

\begin{figure}
    \resizebox{6cm}{!}{\input{figures/svm_convergence_err_approx_comb.pgf}}
    \caption{$\text{Err}_\text{Approx}$ for NS, IJ and IACV in the Smooth SVM problem ($\sigma = 0.5$ and $\lambda = 1$).}
\end{figure}

---

Why IACV?

- Less overall assumptions to satisfy.
- Accuracy before and at convergence. $\mathcolor{red}{\hat{w} \approx w^*}$ not necessary.
- Per-iteration control of approximation.

\pause

*It's not that simple!*

## Main Result

IACV still needs the underlying problem to satisfy assumptions to guarantee accuracy above baseline.

- **Assumption 1:** the LOO Hessian is well-conditioned. \pause
- **Assumption 2:** the LOO Jacobian is bounded along the convergence path. \pause
- **Assumption 3:** the LOO Hessian is $n\gamma-$Lipschitz. \pause

The Smooth SVM problem satisfies Assumptions 2 and 3, though we need to find when Assumption 1 holds.

---

To investigate Assumption 1, we define the condition number. The LOO Hessian is provably positive-definite and symmetric, so we can use this bound.

### Condition Number
For a positive-definite symmetric matrix $A \in \mathbb{R}^{n \times n}$, the condition number $\kappa(A)$ is defined as
\begin{align*}
	\kappa(A) = \|A\|\|A^{-1}\| = \frac{\lambda_{\mathrm{max}}(A)}{\lambda_{\mathrm{min}}(A)}.
\end{align*}

---

We now present the main result.

### Bound for LOO Hessian Condition Number
The condition number of the leave-one-out Hessian for the smooth SVM problem is bound by,
\begin{align*}
	\kappa(\nabla_w^2 L(w; D_{-i}, \sigma)) \leq 1 + \frac{C}{\lambda \sigma} \cdot \frac{1}{2 \sqrt{1 + \left(\frac{m_j}{\sigma}\right)^2}^3},
\end{align*}
for $C = \|\tilde X^T \tilde X\|/(n-1)$ where $m_j = (1 - Y_j w^T X_j)$ is derived from the term

$$\max_j d_j = \max_j \Phi^\prime_M((1 - Y_j w^T X_j)/\sigma)/\sigma.$$

---

\begin{figure}
    \resizebox{10cm}{!}{\input{figures/svm_conv_bound_cond_no.pgf}}
    \caption{Bound along condition numbers across convergence path. ($\sigma = 0.1, \lambda = 1$)}
\end{figure}

---

To use this bound, we can find an optimal $\lambda$ for a given $\sigma$ for a well-conditioned LOO Hessian.
\pause
For a chosen bound $b$ and a given $\sigma$, the chosen $\lambda$ must be at least
$$
	\mathcolor{cyan}{\lambda_b} = \frac{C_f}{(b - 1) \mathcolor{red}{\sigma}} \cdot \frac{1}{2 \sqrt{1 + \left(\frac{m^*}{\sigma}\right)^2}^3},
$$
for a well-conditioned LOO Hessian. Here, $C_f = \|X^T X\|/(n-1)$ and $m^*$ is a small constant close to $0$.

---

::: columns
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_bound_cond_no_lbd_4.pgf}}
\end{figure}
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_bound_cond_no_lbd_100.pgf}}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_bound_cond_no_lbd_10.pgf}}
\end{figure}
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_bound_cond_no_lbd_400.pgf}}
\end{figure}
::::
:::

---

Does the bound work in improving IACV accuracy?

\pause

**Yes!** This is an experiment run for $\sigma = 1 \times 10^{-10}$, picking an optimistic bound of $b = 1 \times 10^{10}$ when picking $\mathcolor{cyan}{\lambda_b}$.

::: columns
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_convergence_err_approx_fail.pgf}}
	\caption{$\text{Err}_\text{Approx}$ when choosing $\lambda = 1$.}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_convergence_err_approx_controlled.pgf}}
	\caption{$\text{Err}_\text{Approx}$ when choosing $\lambda = \mathcolor{cyan}{\lambda_b}$.}
\end{figure}
::::
:::

---

What do the condition numbers look like?

::: columns
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_poor_cond_no.pgf}}
	\caption{Condition number when choosing $\lambda = 1$.}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/svm_controlled_cond_no.pgf}}
	\caption{Condition number when choosing $\lambda = \mathcolor{cyan}{\lambda_b}$.}
\end{figure}
::::
:::

---

Does choosing $\mathcolor{cyan}{\lambda_b}$ impact the overall quality of the model?

\pause

Only slightly.

\begin{figure}
	\resizebox{6.5cm}{!}{\input{figures/svmtest_logreg_sigma_accuracy_comb.pgf}}
	\caption{Accuracy for $\sigma = 0.1$ when picking $\lambda = 1$ and $\lambda = \mathcolor{cyan}{\lambda_b}$.}
\end{figure}

---

To affirm the theory, we run a sensitivity study for both $\sigma$ and $\lambda$. For the tests, we keep $\lambda = 1$ and $\sigma = 0.25$ fixed respectively.

\pause
::: columns
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/svmtest_sigma_err_approx_mod.pgf}}
	\caption{$\text{Err}_\text{Approx}$ for varying $\sigma$.}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/svmtest_sigma_err_cv_mod.pgf}}
	\caption{$\text{Err}_\text{CV}$ for varying $\sigma$.}
\end{figure}
::::
:::


---

::: columns
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/svmtest_lambda_err_approx_mod.pgf}}
	\caption{$\text{Err}_\text{Approx}$ for varying $\lambda$.}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/svmtest_lambda_err_cv_mod.pgf}}
	\caption{$\text{Err}_\text{CV}$ for varying $\lambda$.}
\end{figure}
::::
:::

\pause
\begin{figure}
	\resizebox{4.5cm}{!}{\input{figures/svmtest_lambda_cond_no_mod.pgf}}
\end{figure}

## Python Implementation

How do we use it?

\pause
::: columns
:::: column
\begin{figure}
	\includegraphics[scale=0.65]{figures/IACV_init.png}
\end{figure}
::::
:::: column
\pause
\begin{figure}
	\includegraphics[scale=0.65]{figures/IACV_usage.png}
\end{figure}
::::
:::

---

The underlying implementation uses JAX, making liberal use of `vmap` and `jit` to speed up runtime.

\begin{figure}
	\resizebox{5.5cm}{!}{\input{figures/smoothsvm_cv_benchmark.pgf}}
\end{figure}

\pause
\begin{align*}
    \tilde{\theta}^{(k)}_{-i} &= \tilde{\theta}^{(k-1)}_{-i} - \alpha_k\left(\nbTh \Rreg(\mathcolor{red}{\hat{\theta}^{(k-1)}}; D_{S_k \setminus i}) + \nbTh^2 \Rreg(\mathcolor{red}{\hat{\theta}^{(k-1)}}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)\right)
\end{align*}

---

Improves on existing implementation by an order of magnitude.

\begin{figure}
	\resizebox{13cm}{!}{\input{figures/IACV_runtime.pgf}}
\end{figure}

\begin{figure}
	\includegraphics[scale=0.65]{figures/og_runtime.png}
\end{figure}

---

## Kernelising Smooth SVM

We can kernelise the smooth SVM by substituting $w = \phi(X) \cdot u$ for $u \in \mathbb{R}^n$,
$$
    L(u; D, \sigma) = \frac{\lambda}{2} u^T G u + \frac{1}{n} \sum_{i=1}^n \psi_M(Y_i G_i u; \sigma)
$$
where $G$ is the Gram matrix. \pause

This gives us non-linearity through the feature map $\phi: \mathbb{R}^p \to \mathbb{R}^d$ where $d > p$. **How does kernelisation impact IACV?**

---

The bound for the condition number in this case is,
$$
    \kappa(\nabla_u^2 L(u; D_{-i}, \sigma)) \leq \frac{\|\tilde G^{-1}\| \left(\lambda\|\tilde G\| + C_k B_k\right)}{\lambda}
$$
where $C_k = \|\tilde G^T \tilde G\|/(n - 1)$ and $B_k = 1/\left(2\sigma \sqrt{1 + (\frac{m^*}{\sigma})^2}^3\right)$.

\pause
Assumptions 2 and 3 hold similar to the linear SVM case.

---

The condition number is much less sensitive to change based on $\sigma$ and depends mostly on $\lambda$.

::: columns
:::: column
\begin{figure}
	\includegraphics[scale=0.65]{figures/kernel_sigma_cond_no.png}
	\caption{Kernel SVM.}
\end{figure}
::::
:::: column
\begin{figure}
	\includegraphics[scale=0.65]{figures/linear_sigma_cond_no.png}
	\caption{Linear SVM.}
\end{figure}
::::
:::

---

::: columns
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/kernel_svmtest_sigma_err_approx_mod.pgf}}
	\caption{$\text{Err}_\text{Approx}$ for varying $\sigma$.}
\end{figure}
::::
:::: column
\begin{figure}
	\resizebox{5cm}{!}{\input{figures/kernel_svmtest_sigma_err_cv_mod.pgf}}
	\caption{$\text{Err}_\text{CV}$ for varying $\sigma$.}
\end{figure}
::::
:::

---

::: columns
:::: column
\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_approx_outlier.png}
	\caption{$\text{Err}_\text{Approx}$ for varying $\lambda$.}
\end{figure}
::::
:::: column
\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_cv_outlier.png}
	\caption{$\text{Err}_\text{CV}$ for varying $\lambda$.}
\end{figure}
::::
:::

\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_cond_no.png}
\end{figure}

---

::: columns
:::: column
\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_approx_norm.png}
	\caption{$\text{Err}_\text{Approx}$ for varying $\lambda$.}
\end{figure}
::::
:::: column
\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_cv_norm.png}
	\caption{$\text{Err}_\text{CV}$ for varying $\lambda$.}
\end{figure}
::::
:::

\begin{figure}
	\includegraphics[scale=0.5]{figures/kernel_lambda_cond_no_norm.png}
\end{figure}

# Conclusion & Further Work

Unfortunately, the original goal of applications to high dimensional problems was not fruitful.

In this thesis, we've explored ACV (specifically IACV) in a setting where an explicit Hessian does not exist without smoothing by applying it to Smooth SVM.

\pause
- Analysis of IACV on similar smoothed problems
	- Smoothed ReLU
- IACV in a non-convex setting
- Is IACV is high dimensions feasible?

# Bibliography

\bibliographystyle{alpha}
\bibliography{refs.bib}
