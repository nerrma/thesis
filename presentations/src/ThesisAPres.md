---
header-includes: |
	\usepackage{amsmath}
	 \usepackage{fancyhdr}
	 \usepackage{physics}
	 \usepackage{hyperref}
	 \usepackage{graphicx}
	\graphicspath{ {./images/} }
	\DeclareMathOperator*{\argmax}{arg\,max}
	\DeclareMathOperator*{\argmin}{arg\,min}
title: "Iterative Approximate Cross Validation in High Dimensions"
author: "Sina Alsharif"
theme: "Frankfurt"
institute: "School of Computer Science and Engineering, University of New South Wales"
topic: "Thesis A"
date: "Thesis A (Term 2, 2023)"
colortheme: "orchid"
fonttheme: "professionalfonts"
toc: true
---

# Background
To set out the notation used throughout this seminar, we can define the Empirical Risk Minimisation (ERM) framework to solve a supervised learning problem. 

\pause

### General Problem Setting
- Input space $\mathcal{X}$ and an output space $\mathcal{Y}$.
- Data is ``generated'' by a *true* distribution $P(\mathcal{X}, \mathcal{Y})$.
- Aim is to find a mapping $h : \mathcal{X} \to \mathcal{Y}$ (called a hypothesis).
- Denote all possible combinations of input and output space as $\mathcal{D} = \{(X, Y) \in (\mathcal{X}, \mathcal{Y})\}$.

## Risk

To measure the error (or loss) we make on a data point, define a function $\ell(h; D_i)$,
\begin{align*}
	\ell(h; D) = \sum_{i=1}^n \ell(h; D_i)
\end{align*}
as the loss for the dataset. Examples of $\ell$ are 0-1 loss (for classification) and a squared error (for regression).

\pause

### Risk
Define the **risk** of a hypothesis for the complete space of inputs and outputs as,
\begin{align*}
	R(h) = \mathbb{E}_{\mathcal{X}, \mathcal{Y}} [\ell(h; \mathcal{D})]
\end{align*}
The optimal hypothesis for the data is,
\begin{align*}
	h_{\mathcal{H}} = \argmin_{h \in \mathcal{H}} R(h)
\end{align*}


## Empirical Risk

As we cannot measure **true** risk, we seek an approximation using the observed data $D$.

### Empirical Risk
We define **empirical risk** of a hypothesis given data $D$ as,
\begin{align*}
	R_{\text{emp}}(h; D) = \frac{1}{n} \sum_{i=1}^n \ell(h; D_i)
\end{align*}
The optimal hypothesis *given observed data* $D$ is,
\begin{align*}
	h_D = \argmin_{h \in \mathcal{H}} R_{\text{emp}}(h; D)
\end{align*}
where $\mathcal{H}$ is a hypothesis space which a *learning algorithm* picks a hypothesis from. We will describe the parameters which describe $h$ as $\theta \in \mathbb{R}^p$, assuming the case of a Generalised Linear Model (GLM).

## Issues with ERM

By the (weak) law of large numbers $R_{\text{emp}}(h; D) \to R(h; \mathcal{D})$ as $n \to \infty$, so it is reasonable to assume that $h_D$ converges to a minimiser of true risk. \pause However, if we fit the explicit minimiser of empirical risk, we will not always find the minimiser of true risk \footnote{Refer to the bias-variance (or approximation-estimation) tradeoff.}. We can restrict the learning algorithm's ability to fully minimise empirical risk through **regularisation**.

\pause

### Regularised Empirical Risk
To restrict the hypothesis space, we define the formulation of **regularised empirical risk** as,
\begin{align*}
	R_{\text{reg}}(h; D) = R_{\text{emp}}(h; D) + \lambda \pi(\theta)
\end{align*}
Here, $\pi : \mathbb{R}^p \to \mathbb{R}$ is a regulariser and $\lambda \in \mathbb{R}^+$ is a hyper-parameter to control the strength of regularisation. The solution becomes $(h_D)_\lambda = \argmin_{h \in \mathcal{H}_\lambda} R_{\text{emp}}(h; D)$ where $\mathcal{H}_\lambda$ is a restricted hypothesis space.

## Cross Validation
To avoid ``overfitting'' to the observed data (i.e blindly minimising empirical risk), we can attempt to define an approximation of true risk to measure the efficacy of a learned hypothesis.

\pause

The most common way to estimate the true risk of a hypothesis is to run Cross Validation (CV) for a hypothesis . This is where we break the observed data into small subsets to run multiple ``validation'' experiments (training the data on a subset and testing on an unseen subset).

\pause
One of the most effective methods for risk approximation is Leave One Out Cross Validation (LOOCV) \footnote{Arlot \& Celisse (2008)}. This method is computationally expensive as we repeat the learning task $n$ times (where $n$ is the size of the observed data).

# Literature Review

## Approximate CV Methods
To reduce the computational cost of CV (specifically LOOCV), we turn to Approximate Cross Validation (ACV), which attempts to approximate (rather than solve) individual CV experiments.

### LOOCV
The definition of leave-one-out (LOO) regularised empirical risk is,
\begin{align*}
	R_{\text{reg}}(\theta; D_{-j}) = \sum_{i=1,\,i \neq j}^n \ell(\theta; D_i) + \lambda \pi(\theta)
\end{align*}
where we leave out a point with index $j$ for this experiment.

# Preliminary Work
# Future Plans
