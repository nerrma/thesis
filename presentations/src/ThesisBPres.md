---
header-includes: |
	\usepackage{amsmath}
	 \usepackage{fancyhdr}
	 \usepackage{pgfplots}
	 \usepackage{pgf}
	 \usepackage{lmodern}
	 \usepackage{physics}
	 \usepackage{hyperref}
	 \usepackage{graphicx}
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
topic: "Thesis B"
date: "Thesis B (Term 3, 2023)"
colortheme: "orchid"
fonttheme: "professionalfonts"
toc: true
---

# Approximate CV Methods
To reduce the computational cost of CV (specifically LOOCV), we turn to Approximate Cross Validation (ACV), which attempts to approximate (rather than solve) individual CV experiments.

## LOOCV
The definition of leave-one-out (LOO) regularised empirical risk is,
\begin{align*}
	R_{\text{reg}}(\theta; D_{-j}) = \sum_{i=1,\,i \neq j}^n \ell(\theta; D_i) + \lambda \pi(\theta)
\end{align*}
where we leave out a point with index $j$ for this experiment.

## Quick Recap

There are three main methods for ACV we will discuss. 

- Newton Step (``NS'')
- Infinitesimal Jackknife (``IJ'')
- Iterative Approximate Cross Validation (``IACV'')

both NS and IJ are existing methods, and IACV is a new proposed method which we aim to adapt and extend. \pause 

## Newton Step
For discussion, the standard notation we'll use for the NS method is,
\begin{align*}
    \tilde{\theta}^{-i}_{\text{NS}} = \hat{\theta} + \left(H(\hat{\theta}; D) - \nbTh^2 \ell(\hat{\theta}; D_{i})\right)^{-1} \nbTh R_{\text{reg}}(\hat{\theta}; D_i)
\end{align*}

## Infinitesimal Jackknife

The final form derived for the IJ estimator is
\begin{align*}
    \tilde{\theta}^{-i}_{\text{IJ}} = \hat{\theta} + (H(\hat{\theta}; D))^{-1} \nabla_\theta R_{\text{reg}}(\hat{\theta}; D_i)
\end{align*}
where we again make the same assumptions as in NS (loss and regularisation are continuously twice-differentiable, $H$ is invertible). \pause This method has a computational advantage over NS, as we only need to calculate and invert $H(\hat{\theta}; D)$ once, rather than $n$ times.

## Iterative Approximate Cross Validation
Iterative Approximate Cross Validation (IACV) relaxes these assumptions and gives a more general form.

We again solve the main learning task through an iterative method, where the updates are (for GD and SGD)
\begin{align*}
    \hat{\theta}^{(k)} = \hat{\theta}^{(k-1)} - \alpha_k \nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k})
\end{align*}
where $S_k \subseteq [n]$ is a subset of indices and $\alpha_k$ is a learning rate taken for an iteration $k$. For classic GD, $S_k \equiv [n]$ and can be variable for SGD. \pause

The explicit optimisation step LOOCV iterate excluding a point $i$ is defined as,
\begin{align*}
    \hat{\theta}^{(k)}_{-i} = \hat{\theta}^{(k-1)}_{-i} - \alpha_k \nbTh \Rreg(\hat{\theta}^{(k-1)}_{-i}; D_{S_k \setminus i})
\end{align*}
this step is what we aim to approximate.

--- 

We use a second-order Taylor expansion to reduce the computation burden of calculating the Jacobian for each point:
\begin{align*}
    \nbTh \Rreg(\hat{\theta}^{(k-1)}_{-i}; D_{S_k \setminus i}) \approx \nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) + \nbTh^2 \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)
\end{align*}
is the estimate for the Jacobian.

\pause
Therefore, the IACV updates for GD and SGD become,
\begin{align*}
    \tilde{\theta}^{(k)}_{-i} &= \tilde{\theta}^{(k-1)}_{-i} - \alpha_k\left(\nbTh \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) + \nbTh^2 \Rreg(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)\right)
\end{align*}

---

The main difference (from NS and IJ) is that we can define an ACV update rule for proximal gradient descent. \pause If we define a general update rule for LOOCV proximal gradient descent as,
\begin{align*}
    \hat{\theta}^{(k)}_{-i} &= \argmin_{z} \left\{ \frac{1}{2 \alpha_k} \|z - \theta^\prime_{-i} \|_2^2 + \lambda \pi(z) \right\} \\
    &\text{where } \theta^\prime_{-i} = \hat{\theta}^{(k-1)}_{-i} - \alpha_k \nbTh \ell(\hat{\theta}^{(k-1)}_{-i}; D_{S_k \setminus i})
\end{align*}
using similar logic as in GD/SGD on the differentiable part of the regularised risk, we get IACV updates of, \pause
\begin{align*}
    \tilde{\theta}^{(k)}_{-i} &= \argmin_{z} \left\{ \frac{1}{2 \alpha_k} \|z - \theta^\prime_{-i} \|_2^2 + \lambda \pi(z) \right\} \\
    \text{where }& \theta^\prime_{-i} = \tilde{\theta}^{(k-1)}_{-i} - \alpha_k\left(\nbTh \ell(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) + \nbTh^2 \ell(\hat{\theta}^{(k-1)}; D_{S_k \setminus i}) \left(\tilde{\theta}^{(k-1)}_{-i} - \hat{\theta}^{(k-1)}\right)\right)
\end{align*}

---

The time complexities for the operations are as follows,
\begin{center}
	\begin{tabular}{c|c|c}
		& IACV & Exact LOOCV \\
		\hline
		GD & $n(A_p + B_p) + np^2$ & $n^2 A_p + np$ \\
		SGD & $K(A_p + B_p) + np^2$ & $nKA_p + np$ \\
		ProxGD & $n(A_p + B_p + D_p) + np^2$ & $n^2A_p + nD_p + np$ \\
	\end{tabular}
\end{center}
with breakdown as $p$ increases.

# Experimental Work

We experimented with applying IACV to SVM solved through SGD. I recreated the
algorithm shown in Shai Shalev-Schwartz's text which solves soft-margin SVM through SGD.

\begin{figure}
	\centering
	\resizebox{0.5\linewidth}{!}{\input{./figures/sgd_vs_libsvm.pgf}}
\end{figure}

## Applying IACV

Trying to apply IACV to the SVM algorithm led me to looking a *smooth* hinge
loss \href{https://arxiv.org/abs/2103.00233}{[Luo21]}, which allows for the calculation of the Hessian required for IACV. The simplest form representing this is as follows;

\pause
Define
\begin{align*}
    \Phi_m(v) = (1 + v/\sqrt{(1 + v^2)})/2 \\
    \phi_m(v) = 1/(2 \cdot \sqrt{(1 + v^2)}).
\end{align*}

For a final form of
\begin{align*}
    \psi_m(v) = \Phi_m(v/\sigma) \cdot v + \phi_m(v/\sigma) \cdot \sigma,
\end{align*}
where $\sigma$ controls how close the final result is to the original hinge loss
(i.e as $\sigma \to 0$ we approach the classic hinge loss).

---

My implementation gives reasonable results for $\sigma = 0.8$ (this choice of
$\sigma$ will be explained later).
\begin{figure}
	\centering
	\resizebox{0.5\linewidth}{!}{\input{./figures/smoothhinge_comp.pgf}}
\end{figure}

---

After applying IACV, we retrieve the following statistics from the LOOCV iterates.
\begin{figure}
	\centering
	\resizebox{0.5\linewidth}{!}{\input{./figures/smoothhinge_stats.pgf}}
\end{figure}
\pause
Observe that the mean gives no meaningful information, and indicates a numerical issue.

---

After analysing the sum of the weights when leaving a point out, we see
anomalies (shown in purple) which impact the mean, confidence intervals and
overall utility of the cross-validation. This phenomenon becomes more
apparent (more points with numerical issues) as $\sigma \to 0$.
\begin{figure}
	\centering
	\includegraphics[scale=0.525]{smoothhinge_colorbar.png}
\end{figure}


# Back to High Dimensions

Approximate cross validation tends to break down in high dimensions. For IJ and NS, the main problems are

- Time complexity breakdown (especially for Hessian inversion) 
- A breakdown of accuracy

Where a similar theme follows somewhat for IACV.

## Existing Solutions for High Dimensional Problems

The current solutions for IJ and NS are to use the support of the $\ell$-1
solution at each iteration to reduce to the computation cost and increase the
accuracy of the method. For the estimated support $\hat{S}$, we have
\begin{align*}
    \left[\tilde{\theta}^{-i}_{\text{NS}}\right]_j = 
    \begin{cases}
    0 & \text{when } \hat{\theta}_j = 0 \\
    \hat{\theta}_j + \left[\left(H_{\hat{S}}(\hat{\theta}_{\hat{S}}; D) - \nbTh^2 \ell_{\hat{S}}(\hat{\theta}_{\hat{S}}; D_{-i})\right)^{-1} \nbTh \ell_{\hat{S}}(\hat{\theta}_{\hat{S}}; D_i)\right]_j & \text{otherwise}
    \end{cases}
\end{align*}
where we only evaluate the terms in the support.

Similarly, the ``sparse ACV'' updates for IJ becomes,
\begin{align*}
    \left[\tilde{\theta}^{-i}_{\text{IJ}}\right]_j = 
    \begin{cases}
    0 & \text{when } \hat{\theta}_j = 0 \\
    \hat{\theta}_j + \left[\left(H_{\hat{S}}(\hat{\theta}_{\hat{S}}; D)\right)^{-1} \nbTh \ell_{\hat{S}}(\hat{\theta}_{\hat{S}}; D_i)\right]_j & \text{otherwise.}
    \end{cases}
\end{align*}

---

The paper which proposes this solution is
\href{https://arxiv.org/abs/1905.13657}{[SB20]}. I've had a go at recreating the
results in their paper, with mild success. Running LASSO (for linear
regression) for an experiment with $p = 150$ and $n = 50$ as,

\begin{center}
\begin{tabular}{c | c | c}
		\textbf{Method} & $\text{Err}_\text{App}$ & $\text{Err}_\text{LOO}$ \\
		\hline
		Baseline ($\hat{\theta}$) &  4.25 & 109.56 \\
		IJ (sparse) & 4.16 & 0.35 \\
		NS (sparse) & 4.20 & 0.62 \\
		IACV & 12.3 & 54.8 \\
	\end{tabular}
\end{center}

Where $\text{Err}_{\text{App}} = \frac{1}{n} \sum_{i=1}^n \|\hat\theta_{-i} - \tilde\theta{-i}\|_2^2$ and $\text{Err}_\text{LOO} = \sum_{i=1}^n |\ell(\tilde{\theta}_{-i}) - \ell(\hat{\theta}_{-i})|/\ell(\hat{\theta}_{-i})$.

---

The original paper has errors in the same orders of magnitude.
\begin{figure}
    \centering
    \includegraphics[scale=0.575]{figures/high_dim_error.png}
    \caption{Error rates for smoothed and ``sparse'' (L1) ACV methods on real data \href{https://arxiv.org/abs/1905.13657}{[SB20]}.}
\end{figure}

The next step is to fully recreate the experiments done in the original paper,
however the specific datasets they used are no longer available. \pause Also,
the accuracy of these methods depends highly on the condition $\lambda \geq c
\sqrt{\frac{\log(p)}{n}}$ for some $c$ we pick ($c = n/p$ seems to work fine).


---

They use coordinate descent! 

\pause

This means we cannot directly apply IACV to the same problem, I have tried using
the proximal descent (ISTA) implementation to update IACV iterates alongside the
main coordinate descent to no success.

## Applying Sparse ACV to IACV

I have however, applied the 'sparse' ACV method presented previously to IACV and
the other ACV methods to a logistic lasso task, with $p = 150$ and $n = 50$, solved through ISTA.
\pause
\begin{figure}
	\centering
	\resizebox{0.5\linewidth}{!}{\input{./figures/sparse_err_approx_150.pgf}}
\end{figure}

---

The percentage LOO error is as follows,
\begin{figure}
	\centering
	\resizebox{0.5\linewidth}{!}{\input{./figures/sparse_err_loo.pgf}}
\end{figure}

---

The runtime looks something like this.
\begin{figure}
	\centering
	\resizebox{0.8\linewidth}{!}{\input{./figures/sparse_combined_runtime.pgf}}
\end{figure}

	
# Future Plans

My plans are to stick with improving IACV for high dimensions, since that seems
to have the most ground for improvement and exploration. Things I would like to
get done in the upcoming break and Thesis C are,

\begin{itemize}
	\item General code clean up. 
	\item Experiment with different problem settings for IACV. 
	\item Recreate the Gisette experiment in the Broderick paper (only recently found usable data for this). 
	\item Modify the existing experiment code for a speed-up.
	\item Look into the theory - why is IACV better at approximating coefficients rather than the CV loss?
	\item Is there an IACV-like method for coordinate descent? 
\end{itemize}
