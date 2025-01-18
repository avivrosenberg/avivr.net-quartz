---
aliases: 
tags:
  - machine-learning
related: 
date-created: "[[2024-12-06]]"
date-modified: "[[2025-01-12]]"
date: 2019-10-24
draft: true
enableToc: true
---

# Another look at bias and variance (working title)

The bias-variance tradeoff is a very well-known concept in machine learning which is used to explain the source of the errors we see in our trained models. I often found discussions of this concept confusing and a bit opaque: explaining the meaning of these terms but not what causes them. Why only bias and variance? are these really the only possible problems?

This post is inspired by a lecture by Dr. Nadav Cohen, which I attended at the Technion in January 2019. He presented three types of possible errors from the perspective of statistical learning, and showed a diagram which really made everything "click" for me. This post is my attempt to explain that viewpoint for myself, and to connect it to the tried and true framework of bias and variance.

In this post, we will cover:

* Basics of supervised learning
* Three types of errors in statistical learning
* The connection to the bias-variance tradeoff

## Supervised learning

Let's start by setting the scene. In the context of supervised learning, we're trying to estimate an unknown function from data, possibly noisy and incomplete. There are three main ingredients: the data, the model and the objective.

> [!TODO]

## The bias-variance tradeoff

Let's imagine that we could sample as many different datasets of the same size as we want. What would happen if we fit our model on each of these new datasets?

We would get a different model each time. These different models would not not exactly agree on the predicted value for the same input $\vec{x}$.
In other words, our fitted model's parameters $\vec{\theta}$ are a function of the randomness in the sampling of the dataset $\mat{S}$.

To see this better, lets abuse our previous notation a bit, and write our model's prediction as $\hat{y}=h(\vec{x};\mat{S})$. This notation "imports" the fitting procedure into the hypothesis $h$, and what we mean by this is "$h$ will use whatever parameters $\vec{\theta}$ are obtained when fitting with dataset $\vec{S}$".

Now let's look at some specific input pair $(\vec{x}, y)$ sampled from $\cset{D}$, and consider what happens to the predicted $\hat{y}$ if we fit our model on multiple datasets.

First, notice that that the average prediction for $\vec{x}$ over all possible datasets is $\E[\rvec{S}]{h(\vec{x};\rvec{S})}$. We write $\E[\rvec{S}]{\cdot}$ to emphasize that the expectation here is over the random variable $\rvec{S}$, each value of which is one dataset $\mat{S}$. The result is the average prediction for one specific input $\vec{x}$, if we were to fit the hypothesis class $h$ over all datasets $\mat{S}\sim\rvec{S}$. Now let's answer two questions.

1. How well does our model predict the true target $y$ for input $\vec{x}$? If we define "good" as having a low squared-error, we can write this as
    $$
    \E[\rvec{S}]{\left( y - h(\vec{x};\rvec{S}) \right)^{2} }.
    $$
    This is the (squared) **bias** of the hypothesis class (or model). It's the average squared error that we see when predicting $y$ from $\vec{x}$ using models fit with different datasets.

2. How much does a prediction for $\vec{x}$, obtained on one specific dataset $\mat{S}$, differ from this average prediction? We can write this as,
    $$
    \E[\rvec{S}]{\left(h(\vec{x};\rvec{S})- \E[]{h(\vec{x};\rvec{S})}\right)^{2}}.
    $$
    This is a the **variance** of this hypothesis class (or model), for a specific $\vec{x}$, and it measures how much our prediction would change if we were to fit on different data.

Imagine that we now use a hypothesis class with a higher number of fitted parameters in $\vec{\theta}$ (capacity). What would happen to the bias and variance?

Assuming a perfect fitting procedure, we'd expect more parameters to produce a better fit to whatever dataset the model is fitted on. So,
* The bias would decrease: any one of the fitted models' predictions in the expectation would be closer the true $y$.
* The variance would increase: being able to fit better to each specific dataset, means that the models trained on different datasets can be more different from each other. They might therefore "agree less" on the prediction for any specific $\vec{x}$, which might not have been part of their training. This means that individual predictions could be further from the average prediction, hence high variance.

Such a low-bias but high-variance model is known as being **overfit**, because while it might produce good predictions for the data it was trained on, it may produce far-from-average predictions on unseen data.

Similarly, when decreasing the model's capacity, the bias would generally increase while the variance would decrease. A high-bias but low-variance model is said to be **underfit**.

If this variance is high, it means that the prediction of a specific model, trained on a specific dataset, might be far-off from the average prediction we'd get from different models trained over different datasets.

So generally bias and variance go in different directions when changing the model capacity. This tension is known as the bias-variance tradeoff. Ideally, we want to chose a model and set its capacity such that we find a good tradeoff: sufficiently low bias without too much variance.

## Statistical learning view point

### Optimization ("training")

The model is **trained** by updating its parameters to improve its performance on some data.

We wish to find a parametrization $\vec{w}^\ast$ of our model $h^{\ast}_{\mathcal{S}}(\vec{x};\vec{w}^\ast)\in\mathcal{H}$ such that
$$
h^{\ast}_{\mathcal{S}} = \arg\min_{h\in\mathcal{H}} L_{\mathcal{S}}(h)
$$

Usual approach: descent-based optimization (will be covered in next lecture)
$$
\vec{w}_{k+1} \leftarrow \vec{w}_{k} - \eta_k \vec{d}_{k}
$$

Where $\vec{d}_k$ is a **descent direction** and $\eta_k$ is the step size at step $k$.

Most common choice for $\vec{d}_k$?

Gradient descent: $\vec{d}_k = \nabla_{\vec{w}_{k}} L(\vec{w}_{k})$

Will we find the optimal solution, $\vec{w}^\ast$, though?

Generally the loss is non-convex and we have no guarantee of converging to the global optimum! This leads to **Optimization Error**.
![[dl-tut2-sgd2d_2.png]]

Gradient descent (GD) has the advantage of being extremely simple to apply, and requires only first-order derivatives.

However it provides no guarantees for convergence on non-convex objectives.

Later in the course we'll learn about variants of GD which perform well in practice even on the non-convex objectives we'll face in deep learning.

### Generalization and Expressiveness

We've talked about optimization error.

What are **other** sources of errors in our learning approach (besides optimization)?

We train to minimize our empirical loss $L_\mathcal{S}$ instead of the population loss $L_\mathcal{D}$ $\Rightarrow$ **Generalization Error**

We only consider a limited set of possible functions, $\mathcal{H}$ $\Rightarrow$ **Approximation Error**

![[dl-tut2-error_types.png|width=900]]

How do we mitigate these errors in the practice of machine learning?

* Optimization error: Mini batches; GD variants like stochastic gradient, Momentum, Adam, LR scheduling, etc (we'll see later in the course).

* Generalization error: Get more data; get data which better represents $\mathcal{D}$; train-test splits; cross validation; early stopping; regularization.

* Approximation error: Use a powerful hypothesis class (e.g. DNN); more parameters; "inductive bias" - tailoring the model to the domain (e.g. CNN for images).

## References

This post is based on a tutorial I created for the deep learning course I taught at the Technion Computer Science faculty between 2019 and 2022.

Some images in this post were taken and/or adapted from the following sources:
* MartinThoma [CC0], via Wikimedia Commons <https://commons.wikimedia.org/wiki/File>:Perceptron-unit.svg
* Dr. Nadav Cohen, <http://www.cohennadav.com/files/icermw19_slides.pdf>
* Fundamentals of Deep Learning, Nikhil Buduma, Oreilly 2017
