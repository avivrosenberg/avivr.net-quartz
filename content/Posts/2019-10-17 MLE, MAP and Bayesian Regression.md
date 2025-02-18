---
title: MLE, MAP and Bayesian Regression
aliases: 
tags:
  - machine-learning
related: 
date-created: "[[2024-12-06]]"
date-modified: "[[2025-02-18]]"
date: 2019-10-17
enableToc: true
---

$$
\newcommand{\1}[1]{\mathbb{1}\left[{#1}\right]}
\newcommand{\DKL}[2]{\mathcal{D}_{\text{KL}}\left(#1\,\Vert\, #2\right)}
\newcommand{\DO}[1]{\mathrm{do}\left(#1\right)}
\newcommand{\E}[2][]{\mathbb{E}_{#1}\left[#2\right]}
\newcommand{\OneN}[0]{\frac{1}{N}\mathbf{1}_{N}}
\newcommand{\OneT}[0]{\frac{1}{T}\mathbf{1}_{T}}
\newcommand{\R}[0]{\mathbb{R}}
\newcommand{\T}[0]{{^\top}}
\newcommand{\Var}[1]{\mathrm{Var}\left[#1\right]}
\newcommand{\abs}[1]{\left\lvert#1\right\rvert}
\newcommand{\bb}[1]{\boldsymbol{#1}}
\newcommand{\cset}[1]{\mathcal {#1}}
\newcommand{\diag}{\mathop{\mathrm {diag}}}
\newcommand{\given}[0]{\middle\vert}
\newcommand{\grad}[0]{\nabla}
\newcommand{\indep}{\perp \!\!\! \perp}
\newcommand{\ip}[3]{\left<#1,#2\right>_{#3}}
\newcommand{\mat}[1]{\boldsymbol {#1}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\rvar}[1]{\mathrm {#1}}
\newcommand{\rvec}[1]{\boldsymbol{\mathrm{#1}}}
\newcommand{\setof}[1]{\left\{#1\right\}}
\newcommand{\trace}[1]{\mathrm{tr}\left(#1\right)}
\renewcommand{\Pr}[1]{\mathbb{P}\left[#1\right]}
\renewcommand{\exp}[1]{\mathrm{exp}\left(#1\right)}
\renewcommand{\vec}[1]{\boldsymbol {#1}}
$$

# MLE, MAP and Bayesian Regression

Regression is arguably one of the most fundamental tasks in machine learning (and statistics).

Given some datapoints and corresponding target values, we want to find a function that "predicts" target values for new input datapoints. For example, our data might contain features about houses (square area, location, number of floors, whether there's a view, etc) and the target might be the price of the house. A regression model could be fit on such data, and produce a predicted price given the features of a new house.

I've encountered this topic many times in different contexts, with different terminology, mathematical tools and assumptions used in order to formalize it.
Usually terms like "likelihood" and "posterior" are used. There are also some "well known" facts (to those who know them well) which are often mentioned in passing as trivial or easy to show.

This post is my attempt to explain this topic to myself, and to highlight the connections between various ways in which regression problems are often formulated. Specifically, it will cover:

* The basics of supervised learning
* A probabilistic view of regression
* MLE vs MAP
* Bayesian regression, and how it compares to MLE and MAP
* How the squared error loss is a consequence of MLE
* Why L2 regularization is equivalent to a Gaussian prior

## Supervised learning

Let's start by setting the scene. In the context of supervised learning, we're trying to estimate an unknown function from labelled data, possibly noisy and incomplete. There are three main ingredients: the data, the model and the objective.

### Data

We are given a dataset of $N$ labelled samples $\mat{S} = \setof{ (\vec{x}^{(i)},y^{(i)}) }_{i=1}^N$, where
* $\vec{x}^{(i)} = \left(x^{(i)}_1, \dots, x^{(i)}_n\right) \in \mathcal{X}\subseteq \mathbb{R}^{n}$ is a sample, or feature vector.
* $y^{(i)} \in \mathcal{Y}$ is the label. Here we'll focus on simple regression tasks, where we'll usually have $\mathcal{Y}=\mathbb{R}$ for simple univariate regression.

All we're assuming here that each labeled sample was **independently** drawn from some unknown joint distribution $\mathcal{D}$ over $\mathcal{X}\times\mathcal{Y}$, i.e. $(\vec{x}^{(i)},y^{(i)})\sim\mathcal{D}$. This joint distribution represents the true data-generating process, to which we have no access; all we see are samples from it.

Note also that from this perspective, our given dataset $\mat{S}$ is therefore an instance of a random variable $\rvec{S}$.

### Model

Our model is a parametrized function $h(\vec{x};\vec{\theta}) \in \cset{H}$, where $\mathcal{H}\subseteq \mathcal{Y}^{\mathcal{X}}$ is a family of functions known as the **hypothesis class** ($\cset{Y}^{\cset{X}}$ denotes the set of all functions from $\mathcal{X}$ to $\mathcal{Y}$). The functions in $\cset{H}$ differ only by their parameters $\vec{\theta}\in \mathbb{R}^{m}$, hence they are a "family".

For example, here are a few very common hypothesis classes.
* Linear: a simple affine transformation of the features.
    $$
    \mathcal{H} = \left\{ h : h(\vec{x}) = \vec{w}\T \vec{x}+b \right\}
    $$
    Note that the model parameters are $\vec{\theta}=(\vec{w},b)$.

* Linear with fixed basis functions: as above, but where an arbitrary (but fixed) transformation $\phi:\mathbb{R}^{d}\to \mathbb{R}^{d'}$ is applied to the features. This transformation can be thought of as a feature extraction or pre-processing step.
    $$
    \mathcal{H} = \left\{ h : h(\vec{x}) = \vec{w}\T \phi(\vec{x})+b \right\}
    $$
    Note that this is still a linear model, since it's linear in the model parameters (weights), but it's far more expressive since the relationship between the input features and output can be arbitrarily complex.

* Perceptron: a non-linear function applied the output of an affine transformation.
    $$
    \mathcal{H} = \left\{ h: h(\vec{x}) = \varphi(\vec{w}\T\vec{x}+b) \right\}
    $$
    Here $\varphi(\cdot)$ is some nonlinear function, for example a sigmoid. This hypothesis class is used e.g. in logistic regression.
* Neural network: a composition of $L$ layers, each with multiple perceptron models.
     $$
     \mathcal{H} = \left\{ h:
     h(\vec{x})= \varphi \left(
     \mat{W}_L \varphi \left(
     \cdots \varphi \left( \mat{W}_1 \vec{x} + \vec{b}_1 \right)
     \cdots \right) + \vec{b}_L
     \right)
     \right\}
     $$
    Where the $\mat{W}_{l},\ \vec{b}_{l}$ are weight matrices and bias vectors, respectively. In this context, the humble perceptron model is called a **neuron** instead (due to the biologically inspired origin of this model). Note that each row of each weight matrix corresponds to a separate neuron.

In any case, our model outputs a prediction, which we'll denote as $\hat{y}=h(\vec{x};\vec{\theta})$.
This notation emphasizes that $h$ is a *parametrized* function. In the context of fitting (or training) the model, we'll think of $h$ mainly as a function of the parameters, $\vec{\theta}$. In the context of inference (using the model to make predictions), we'll think of $\vec{\theta}$ as fixed and think of $h$ as a function of the input data, $\vec{x}$.

The number of parameters in the hypothesis class is sometimes referred to as the model's "capacity". Intuitively, the greater the capacity, the better it can fit the given data.

### Objective

We'd like our model's outputs for the given input features to be "close" in some way to given target values. The objective function is what measures this "closeness", so we want our model's predictions to maximize it with respect to the given data. In machine learning, the objective is usually defined as a loss, i.e. a function we want to minimize.

A **pointwise** loss is some function $\ell:\mathcal{Y}\times\mathcal{Y}\to\mathbb{R}_{\geq 0}$: it "compares" a predicted value $\hat{y}$ to the ground truth target value $y$. The most common example in the context of regression is the squared error: $\ell(y,\hat{y})=(y-\hat{y})^2$. We'll see later why this is a very natural choice to make.

Ideally, we would like to find a predictor function (hypothesis) $h\in \cset{H}: \mathcal{X}\to\mathcal{Y}$ which minimizes,
$$
L_{\mathcal{D}}(\vec{\theta}) = \E[(\vec{x},y)\sim\mathcal{D}]{\ell(y, h(\vec{x};\vec{\theta})}.
$$
This is known as the **population** loss or **out-of-sample** loss. Notice that it's not computed on the dataset, but over the true distribution. This is actually a **deterministic** quantity: it's the expectation of a random variable.

But can we actually solve this problem, i.e. find the minimizer for $L_{\mathcal{D}}(h)$?

No, because the joint-distribution $\mathcal{D}$ is unknown. Even if it were known, we'd need to compute the expectation over all possible data points, which is generally intractable (except in special cases).

Instead, given our training set $\mat{S} = \setof{ (\vec{x}^{(i)},y^{(i)}) }_{i=1}^N \sim \mathcal{D}$, we define an **empirical (in-sample) loss** $L_{\mat{S}}(\vec{\theta})$, as the measure of how well a function $h\in\mathcal{H}$ fits the dataset $\mat{S}$. Using our point-wise loss, we can define the loss over the dataset as simply the empirical mean,
$$
L_{\mat{S}}(\vec{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \ell\left(h(\vec{x}^{(i)};\vec{\theta}), y^{(i)}\right).
$$

Is the empirical loss also a deterministic quantity?

No, because it depends on the randomness of the dataset sampling. In other words, it's computed only on a single realization $\mat{S}$ of a random variable $\rvec{S}$. A different sampling of $\mat{S}$ would give us a different optimal $\vec{\theta}$.

### Regularization

The objective may be augmented with an additional term $R(\vec{\theta})$ which depends entirely on the model. This is known as regularization.

Regularization is added in order to "encourage" the model to have some additional properties. A common reason to add it is to prevent our model from being too dependent on the specific data in $\mat{S}$, which is known as overfitting or having a high-variance model. Another way to view regularization, is to think of it as a way to encode our prior (i.e., before seeing any data) beliefs about the model.

Two very common regularization terms are,
* L2, i.e. $R(\vec{\theta})=\norm{\vec{\theta}}_{2}^{2}$. Adding this results in what's known as Ridge regression. This penalizes large magnitudes of the parameter values. We'll see why it makes sense later on.
* L1, i.e. $R(\vec{\theta})=\norm{\vec{\theta}}_{1}$, which produces Lasso regression. This also penalizes large magnitudes, but in addition promotes parameter sparsity (but that's for another post).

Choosing the squared error pointwise loss and adding L2 regularization would give us the following objective to minimize,
$$
\tilde{L}_{\mat{S}}(\vec{\theta}) = \frac{1}{N} \sum_{i=1}^{N} \left(h(\vec{x}^{(i)};\vec{\theta}) - y^{(i)}\right)^{2} + \lambda \norm{\vec{\theta}}_{2}^{2},
$$
where $\lambda$ is a constant which determines the *regularization strength* (i.e., how much we want to penalize large norms in the objective). We'll see more about $\lambda$ later.

## The probabilistic view

Let's go a bit deeper and consider the actual assumptions being made to formulate the supervised learning problem.

Suppose we further assume that given an input feature $\vec{x}$, the label is generated by a deterministic function with some additive zero-mean Gaussian noise:
$$
y^{(i)}=f^{*}(\vec{x}^{(i)})+\eta^{(i)},
$$
where $f^{*}(\vec{x})$ is the ground-truth deterministic mapping from feature to target, and $\eta^{(i)} \sim \cset{N}(0,\sigma^{2}_{n})$ are independent noise samples. This is a very common assumption, and although it costs us some generality, it is reasonable for many real-world applications, and will be useful for our analysis.

Now, instead of just predicting one target value $\hat{y}$, we would actually like to know the conditional distribution $p(y | \vec{x})$. If we knew it, we could not only predict $y$ (e.g. by choosing the mean as the prediction), but also do other useful things such as estimate the level of uncertainty in our prediction.

To estimate the unknown conditional distribution, we can parametrize it as $p(y | \vec{x}, \vec{\theta})$, and use our assumption about the data generating process, which would give us
$$
p(y\vert\vec{x},\vec{\theta}, \sigma_{n}^{2})
= h(\vec{x};\vec{\theta})+\cset{N}(0,\sigma_{n}^{2})
= \cset{N}(h(\vec{x};\vec{\theta}),\sigma_{n}^{2}),
$$
which means that the conditional probability distribution is Gaussian, centered around our model's prediction, and with a fixed variance $\sigma_{n}^{2}$ (which does not depend on $\vec{x}$, as opposed to the mean). Note that now we also need to estimate the variance $\sigma_{n}^{2}$ from the data.

Given these assumptions, how can we estimate the distribution of interest, $p(y|\vec{x})$?

We'll look at three different options: maximum likelihood estimation (MLE), maximum a posteriori (MAP), and Bayesian inference.

### Maximum likelihood estimation

The parametrized distribution $p(y\vert\vec{x},\vec{\theta}, \sigma_{n}^{2})$ is called a likelihood. We can use our dataset $\mat{S}$ to obtain the $\vec{\theta}$ which maximizes the probability of observing the data that we have. The set of model parameters which maximize the likelihood of our can be expressed as
$$
\hat{\vec{\theta}}_{\text{MLE}}=\arg\max_{\vec{\theta}} \prod_{i=1}^{N} p(y^{(i)}|\vec{x}^{(i)},\vec{\theta},\sigma_{n}^{2}).
$$
We're using the fact that the dataset was sampled i.i.d., to write its total probability as a simple product.

In practice, working with products is inconvenient. The common trick to dealing with such cases is to maximize the logarithm of the total likelihood instead: because log is a monotonic function, we'll obtain the same maximizer, $\hat{\vec{\theta}}_{\text{MLE}}$, and the log of a product is a sum, making it easier to work with. So we have,
$$
\begin{align}
\hat{\vec{\theta}}_{\text{MLE}}
=\arg\max_{\vec{\theta}} \prod_{i=1}^{N} p(y^{(i)}|\vec{x}^{(i)},\vec{\theta},\sigma_{n}^{2})
=\arg\max_{\vec{\theta}} \sum_{i=1}^{N} \ln \left(  p(y^{(i)}|\vec{x}^{(i)},\vec{\theta},\sigma_{n}^{2})  \right).
\end{align}
$$
To proceed, we'll plug in our assumption about the specific form of the likelihood: it's a Gaussian distribution, i.e.
$$
p(y|\vec{x},\vec{\theta},\vec{\sigma_{n}^{2}})=
\frac{1}{\sqrt{2\pi \sigma_{n}^{2}}}
\exp{-\frac{1}{2 \sigma_{n}^{2}}\left( y -h(\vec{x};\vec{\theta}) \right)^{2}}.
$$

Plugging-in the above and taking the log, we're left with
$$
\begin{align}
\hat{\vec{\theta}}_{\text{MLE}}
&=\arg\max_{\vec{\theta}} \sum_{i=1}^{N} \left\{
-\frac{1}{2}\ln(2\pi \sigma_{n}^{2})
- \frac{1}{2\sigma_{n}^{2}} \left(y^{(i)}-h(\vec{x}^{(i)};\vec{\theta})\right)^{2}
\right\} \\
&=\arg\min_{\vec{\theta}} \left\{ \sum_{i=1}^{N} \left( y^{(i)}-h(\vec{x}^{(i)};\vec{\theta}) \right)^{2} \right\}.
\end{align}
$$
To get the last equation, we dropped the first term and the $1/2\sigma_{n}^{2}$ factor, since both don't change the maximum w.r.t. $\vec{\theta}$, and then flipped the sign to write the maximization as minimization.

A sharp-eyed reader will immediately notice that we're left with a simple minimization of the squared error pointwise loss over our data.

This shows why the squared error pointwise loss is such a natural choice: it **arises** directly from likelihood maximization, if we're willing to make the fairly reasonable assumption that the data generating being a deterministic function of $\vec{x}$ with additive Gaussian noise.

We can now also empirically estimate the variance $\sigma_{n}^{2}$:
$$
\widehat{\sigma_{n}^{2}}=\frac{1}{N} \sum_{i=1}^{N} \left( y^{(i)} - h(\vec{x}^{(i)};\hat{\vec{\theta}}_{\text{MLE}}) \right) ^{2},
$$
which is the standard variance estimator for our data, computed around the mean given by our maximum-likelihood model.

We thereby achieved our goal: now, given a new input $\vec{x}$, we have an estimate for the full distribution of $y$ instead of just being able to predict a single value:
$$
p(y|\vec{x})=\cset{N}(h(\vec{x};\hat{\vec{\theta}}_{\text{MLE}}), \widehat{\sigma_{n}^{2}}).
$$

### Maximum a posteriori estimation (MAP)

One thing that might seem un-intuitive above the MLE approach, is that we're maximizing the probability of observing the data we have, given some model parameters. But in practice, what we face with supervised learning is the other way around: we are *given* the dataset $\mat{S}$, and want to estimate the most probable model parameters $\vec{\theta}$.

In other words, it seems to make more sense to maximize the probability of the parameters given the data, i.e. $p(\vec{\theta}|\mat{S})$. Such an estimate is known as maximum a posteriori, or MAP.

In the Bayesian terminology, $p(\mat{S}|\vec{\theta})$ is the **likelihood**, which represents the probability of the data given the model parameters. The parameters in turn are assumed to be distributed according to a **prior** $p(\vec{\theta})$, and then $p(\vec{\theta}|\mat{S})$ is called the **posterior**. The terms "prior" and "posterior" here refer to whether the distribution of the model parameters is computed "before" or "after" seeing the data. To complete the set of distributions, there's also $p(\mat{S})$ which is called the **evidence**.

The relation between these probability distributions is given by Bayes' rule,
$$
\underbrace{p(\vec{\theta}|\mat{S})
\vphantom{\frac{p(S)}{p(S)}}}_{\text{posterior}}
=
\underbrace{p(\vec{\theta})
\vphantom{\frac{p(S)}{p(S)}}}_{\text{prior}}
~~\cdot
\underbrace{
\frac{p(\mat{S}|\vec{\theta})}{p(\mat{S})}
}_{\text{update factor}},
$$
and a MAP estimate seeks the model parameters which maximize this posterior. Because the $p(\mat{S})$ does not affect the location of the maximum, we can ignore it, and write
$$
\hat{\vec{\theta}}_{\text{MAP}}=\arg\max_{\vec{\theta}}
\left\{ p(\vec{\theta}) \cdot p(\mat{S}|\vec{\theta}) \right\},
$$
which is very similar to the MLE problem above -- it's just multiplied by the prior.

At this point we need to introduce another assumption so that we can proceed with using the prior. We'll assume that the model parameters come from a zero-mean multivariate Gaussian distribution: $p(\vec{\theta})=\cset{N}(\vec{0},\sigma_{\theta}^{2}\mat{I})$, where $\sigma_{\theta}^{2}$ is another unknown variance which we'll treat as a hyperparameter:
$$
p(\vec{\theta}|\sigma_{\theta}^{2})=
\frac{1}{\sqrt{ (2\pi)^{m} \sigma_{\theta}^{2} }}
\exp{-\frac{1}{2 \sigma_{\theta}^{2}} \norm{\vec{\theta}}_{2}^{2}}.
$$
This assumption is not very restrictive, yet leads to an interesting result which we'll see shortly. We can think of $\sigma^{2}_{\theta}$ as the (inverse) **strength** of the prior: the larger this variance, the more uniform the distribution becomes, so the weaker our prior. Choosing $\sigma_{\theta}^{2}$ to be very small, creates a narrow distribution around zero, indicating we're confident about this prior (that the model parameters should all be very small).

Another advantage of this assumption from an analysis perspective, is that the resulting maximization objective $p(\vec{\theta})\cdot p(\mat{S}|\vec{\theta})$ is then also Gaussian, being a product of two Gaussian distributions. The posterior $p(\vec{\theta}|\mat{S})$ is itself also Gaussian because $\mat{S}$ is treated as a constant, so $p(\mat{S})$ is also a constant (though unknown).

We can now plug this prior into the MAP estimate together with our likelihood from the previous section. We'll also condition on the variance terms to treat them as constants. This gives us,
$$
\begin{align}
\hat{\vec{\theta}}_{\text{MAP}}
=\arg\max_{\vec{\theta}} \left\{
\underbrace{p(\vec{\theta}|\sigma_{\theta}^{2})
\vphantom{\prod_{i=1}^{N}}}_{\text{prior}}
\cdot
\underbrace{ \prod_{i=1}^{N} p(y^{(i)}|\vec{x}^{(i)},\vec{\theta},\sigma_{n}^{2}) }_{ \text{likelihood}}
\right\}
\end{align}.
$$

This expression may seem a bit unwieldy, until we remember that both these distributions are just Gaussians, and that we have a neat trick which breaks these products into sums while simultaneously dropping exponents. By taking the log and removing constant terms from the maximization, we're left with,
$$
\hat{\vec{\theta}}_{\text{MAP}}=
\arg\max_{\vec{\theta}} \left\{
-\frac{1}{2 \sigma_{\theta}^{2}} \norm{\vec{\theta}}^{2}
- \frac{1}{2\sigma_{n}^{2}} \sum_{i=1}^{N}
- \left(y^{(i)}-h(\vec{x}^{(i)};\vec{\theta})\right)^{2}
\right\},
$$
which can be re-written equivalently as the minimization,
$$
\hat{\vec{\theta}}_{\text{MAP}}=
\arg\min_{\vec{\theta}}
\left\{
\sum_{i=1}^{N}
\left(y^{(i)}-h(\vec{x}^{(i)};\vec{\theta})\right)^{2}
+\frac{\sigma_{n}^{2}}{\sigma_{\theta}^{2}} \norm{\vec{\theta}}^{2}
\right\}.
$$

If this expression seems familiar, that's because this is nothing but the MLE objective with an L2 regularization term.

The key takeaway here is that adding L2 regularization is exactly equivalent to a Gaussian prior on the model parameters. It turns the MLE into a MAP.

We can also see that the regularization strength is $\lambda={\sigma_{n}^{2}}/{\sigma_{\theta}^{2}}$. This has some rather intuitive interpretations which can guide our choice of the hyperparameter $\lambda$:
* The regularization strength is inversely proportional to the variance of the prior. The more we're confident that our prior is informative (low $\sigma_{\theta}^{2}$), the more regularization we should add (high $\lambda$), and vice versa.
* For a fixed prior-strength ($1/\sigma_{\theta}^{2}$), we should use stronger regularization if the dataset noise is strong (high $\sigma_{n}^{2}$). This corresponds to the notion that regularization prevents overfitting to the noise in the data.

### Bayesian Regression

In both the MLE and MAP approaches, we ended up with a *point estimate* for the model parameters, i.e. a single value ($\hat{\vec{\theta}}_{\text{MLE}}$ or $\hat{\vec{\theta}}_{\text{MAP}}$) which represents the "best" set of model parameters for our chosen hypothesis class.

In these cases, it was "best" in the sense that it corresponds to the location of the maximal value (mode) of either the likelihood $p(\mat{S}|\vec{\theta})$ or the posterior $p(\vec{\theta}|\mat{S})$ distributions. However, choosing the input value that maximizes a distribution means we're **ignoring** most of the information of that distribution.

Consider the posterior $p(\vec{\theta}|\mat{S})$, which we know is also Gaussian (under the assumptions we made about the likelihood and prior), so the mode (maximum) is also the mean. But what about its variance? Intuitively, if this distribution is very narrow (low variance), then choosing a single value $\hat{\vec{\theta}}$ to represent it makes sense. But as the variance increases, other values of $\vec{\theta}$ become more and more likely as well, yet we're still ignoring them and only choosing the value which corresponds to the mean.

A fully Bayesian approach would take the **full** posterior distribution into account, not just its mode. It does this by *marginalizing* over $\vec{\theta}$: i.e., summing over all possible values of $\vec{\theta}$ weighted by their probability.

In the Bayesian case, we estimate the **posterior-predictive** distribution, which in our notation is $p(y\vert \vec{x}, \mat{S})$, i.e. the probability of a possible target $y$ given a new data point $\vec{x}$ and the training data $\mat{S}$. Notice the subtle difference from what we did before: In MLE/MAP we estimated $p(y|\vec{x})$, but the conditioning on $\mat{S}$ in those cases was only through a point estimate $\hat{\vec{\theta}}$.

Estimating the posterior-predictive by marginalizing over $\vec{\theta}$ means computing,
$$
p(y|\vec{x}, \mat{S})=\int p(y|\vec{x},\vec{\theta})p(\vec{\theta}|\mat{S})d \vec{\theta}.
$$
The first term in this integral is the likelihood for a single new datapoint $(\vec{x},y)$, while the second term is the posterior. Under the assumptions above, both are Gaussian. The result of this integral can therefore be computed analytically, though that's beyond the scope of this post. One interesting thing to note about the resulting $p(y|\vec{x}, \mat{S})$ is that its variance can also depend on $\vec{x}$, whereas in the previous cases we saw that $\vec{x}$ only determines the mean.

In the more general case where we don't assume a Gaussian prior and likelihood, there might not be any way to calculate this integral analytically. Instead, numerical methods must be used to approximate it, making it more computationally intensive to produce each prediction. This might be one of the reasons that the fully-Bayesian approach is perhaps less often used in practice.

## Conclusion

The classic machine learning task of regression, often viewed as merely fitting a function and obtaining point predictions, can also be viewed as fitting a probability distribution from which predictions can be sampled.

When taking this view, we need to contend with (at least) three common approaches for fitting the parameters of this distribution. Maximum Likelihood Estimation (MLE), Maximum A Posteriori (MAP), and Bayesian regression. In this post, we saw how these techniques are all interrelated within the same probabilistic framework.

MLE focuses on finding parameters that maximize the likelihood of the observed data, while MAP extends MLE to incorporate prior beliefs about the model parameters. We saw how under some reasonable assumptions, MAP naturally leads to L2 regularization (known as *weight decay* in other contexts). By incorporating a prior, MAP mitigates the tendency of MLE to overfit. Bayesian regression takes this a step further by accounting for the full posterior distribution of the parameters, whereas MLE and MAP only consider a single point on this distribution.

## References

The following sources helped me understand these topics.
* Pattern Recognition and Machine Learning (Bishop, 2006)
* Machine learning: A Probabilistic Perspective (Murphy, 2012)
