---
aliases: 
tags:
  - dl-tutorial
  - machine-learning
related: 
date-created: "[[2024-12-06]]"
date-modified: "[[2025-01-18]]"
title: Logistic regression from scratch
date: 2019-10-31
enableToc: true
---

$$
\newcommand{\mat}[1]{\boldsymbol {#1}}
\newcommand{\mattr}[1]{\boldsymbol {#1}^\top}
\newcommand{\matinv}[1]{\boldsymbol {#1}^{-1}}
\newcommand{\vec}[1]{\boldsymbol {#1}}
\newcommand{\vectr}[1]{\boldsymbol {#1}^\top}
\newcommand{\rvar}[1]{\mathrm {#1}}
\newcommand{\rvec}[1]{\boldsymbol{\mathrm{#1}}}
\newcommand{\diag}{\mathop{\mathrm {diag}}}
\newcommand{\set}[1]{\mathbb {#1}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}
\newcommand{\pderiv}[2]{\frac{\partial #1}{\partial #2}}
\newcommand{\bb}[1]{\boldsymbol{#1}}
\newcommand{\E}[2][]{\mathbb{E}_{#1}\left[#2\right]}
\newcommand{\given}[]{~\middle\vert~}
$$

# Logistic Regression

## Introduction

Logistic regression (LR) is a fundamental algorithm in machine learning and statistics. Despite its name, this is actually a **classification** model, where we're trying to categorize data into two or more groups given some input features. Due to its simplicity, LR is widely used in many applications, especially for binary classification such as detecting whether an email is spam or not.

This post is a hands-on tutorial. We will build logistic regression from scratch, focusing on both the theory and implementation. Along the way, we will derive the logistic regression model based on maximum likelihood estimation.

We'll implement the approach from scratch twice: first using NumPy to understand the underlying mechanics, and then using PyTorch to also leverage automatic differentiation. We'll also extend the binary logistic regression model to handle multiclass problems using the softmax function.

This post should help you will understand the mathematical foundations of logistic regression, how it models probabilities for classification, and the steps involved in implementing it from scratch using NumPy and PyTorch. You will also explore its strengths and limitations in practical applications.
See also my [[2019-10-17 MLE, MAP and Bayesian Regression#Supervised learning|previous post]] for a broader intro about supervised learning and maximum likelihood estimation (which will be relevant here).

> [!info]- Note
> This post is based on materials created by [[About me|me]] for the [CS236781 Deep Learning](https://vistalab-technion.github.io/cs236781/semesters/w22/info/) course at the Technion between Winter 2019 and Spring 2022. To re-use, please provide attribution and link to this page.

## Binary Logistic Regression

Given inputs $\vec{x}^i \in \set{R}^D$ and targets $y^i \in \{0,1\}$, the LR model is
$$
\hat{y} = h(\vec{x}) = \sigma(\vectr{w}\vec{x}+b),
$$
where $\sigma(\vec{z})$ is the **logistic function**,
$$
\sigma(\vec{z}) = \frac{1}{1+e^{-\vec{z}}}.
$$
This function maps the real line onto $[0,1]$.

Notice how the LR model is actually a linear regression model wrapped in a sigmoid, to squash the output into the desired range. From a probabilistic perspective, we're modeling the conditional probability to observe $y=1$ given an input $\vec{x}$, i.e. $\hat{y}=P(\rvar{Y}=1|\rvec{X}=\vec{x})$.

Let's take a look at the logistic function itself:
```python
def logistic(z):
    return 1 / (1 + np.exp(-z))

x = np.arange(-5, 5, .01)
y_hat = logistic(x)

plt.plot(x, y_hat, label='$\sigma(z)$')
plt.grid(True); plt.xlabel('z'); plt.legend(); plt.title('The logistic function');
```

![png|darkmodeinvert](dl-tut2-output_35_0.png)

To fit the model, we'll minimize the **negative log-likelihood** (equivalent to maximizing likelihood, i.e. MLE) of the parameters $\vec{w}$:
$$
\bb{w}^\ast = \mathrm{arg}\max_{ \bb{w}} \prod_{i=1}^n P(y_i | \bb{x}_i;\vec{w}) = \mathrm{arg}\min_{ \bb{w}} \sum_{i=1}^n -\log P(y_i | \bb{x}_i; \vec{w})
$$

We can then define our loss $L(\bb{w})$ as,
$$
\begin{align}
L(\bb{w}) &\triangleq \sum_{i=1}^n -\log P(y_i | \bb{x}_i; \vec{w})\\
&= \sum_{i=1}^n -y_i \log P(y_i=1 | \bb{x}_i; \vec{w}) - (1-y_i) \log P(y_i=0 | \bb{x}_i; \vec{w}) \\
&= \sum_{i=1}^n -y_i \log \hat{y}_i - (1-y_i) \log(1-\hat{y}_i)
\end{align}
$$

The resulting pointwise loss is also known as (binary) **cross-entropy**:
$$
\begin{align}
\ell(y, \hat{y}) &=  - y \log(\hat{y}) - (1-y) \log(1-\hat{y}) \\
\end{align}
$$

The "cross" here is between the distribution of the samples $y^i$ and the distribution of the predictions $\hat{y}^i$.

In this case there's no closed form solution, so we'll need to train the model using some optimization scheme. Since this loss is **convex**, gradient-based approach should lead us to the global optimum.

We can plot this loss function, to convince ourselves of its convexity.

```python
loss_y0 = -np.log(1-y_hat)
loss_y1 = -np.log(y_hat)
plt.plot(y_hat, loss_y0, label='$\ell(y=0,\hat{y})$')
plt.plot(y_hat, loss_y1, label='$\ell(y=1,\hat{y})$')
plt.grid(True); plt.xlabel('$\hat{y}$'); plt.legend(); plt.title('Cross entropy loss');
```

![png|darkmodeinvert](dl-tut2-output_41_0.png)

Next, to apply gradient descent, we'll need to know the gradient of the loss w.r.t. the parameters $\vec{w}$. Let's quickly derive this ourselves.

First, we apply the chain-rule
$$
\pderiv{\ell(y,\hat{y})}{\vec{w}}=\pderiv{\ell}{\hat{y}}\cdot\pderiv{\hat{y}}{z}\cdot\pderiv{z}{\vec{w}},
$$
where $z=\vectr{w}\vec{x}$. Now, just by definition,
$$
\begin{align}
\pderiv{\ell}{\hat{y}}&=-\frac{y}{\hat{y}}+\frac{1-y}{1-\hat{y}}=\frac{\hat{y}-y}{\hat{y}(1-\hat{y})}\\
\pderiv{\hat{y}}{z}&=\frac{e^{-z}}{(1+e^{-z})^2}=
\frac{e^{-z}}{1+e^{-z}}\cdot\frac{1}{1+e^{-z}}=(1-\sigma(z))\sigma(z)=(1-\hat{y})\hat{y}\\
\pderiv{z}{\vec{w}}&=\vec{x}
\end{align}
$$

So we have found that for the cross entropy loss with binary logistic regression, the gradient is
$$
\pderiv{\ell(y,\hat{y})}{\vec{w}}= (\hat{y}-y)\vec{x}.
$$

## Part 1: Binary Logistic Regression with numpy

As a warm up, we'll start by implementing this algorithm and training it from scratch using just `numpy` (and a toy dataset from `sklearn`).

This is a classic and very simple example of implementing and training a machine learning algorithm. By first using only `numpy`, we get to see the barebone details without the help of any fancy machine-learning library.

### Dataset

The `scikit-learn` library comes with a few [toy datasets](http://scikit-learn.org/stable/datasets/index.html#toy-datasets) that are fun to quickly train small models on.

As an example we'll load the Wisconsin-breast cancer database:
* 569 samples of cancer patients
* 30 features: various properties of tumor cells extracted from images
* 2 classes: Tumor is either Benign or Malignant

We'll apply the basic machine learning approach we saw above: binary logistic regression.

```python
import sklearn.datasets

ds_cancer = sklearn.datasets.load_breast_cancer()
feature_names = ds_cancer.feature_names
target_names = ds_cancer.target_names
n_features = len(feature_names)

print(f'{n_features=}')
print(f'feature names: {feature_names}')
print(f'target  names: {target_names}')
```

    n_features=30
    feature names: ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
     'mean smoothness' 'mean compactness' 'mean concavity'
     'mean concave points' 'mean symmetry' 'mean fractal dimension'
     'radius error' 'texture error' 'perimeter error' 'area error'
     'smoothness error' 'compactness error' 'concavity error'
     'concave points error' 'symmetry error' 'fractal dimension error'
     'worst radius' 'worst texture' 'worst perimeter' 'worst area'
     'worst smoothness' 'worst compactness' 'worst concavity'
     'worst concave points' 'worst symmetry' 'worst fractal dimension']
    target  names: ['malignant' 'benign']

```python
X, y = ds_cancer.data, ds_cancer.target
n_samples = len(y)

print(f'X: {X.shape}')
print(f'y: {y.shape}')
```

    X: (569, 30)
    y: (569,)

First we need to handle the data: loading, splitting and processing it. We start by loading it into a `pandas` dataframe and show some samples.
```python
y_names = np.full_like(y, target_names[0].upper(), dtype=target_names.dtype)
y_names[y==1] = target_names[1].upper()

df_cancer = pd.DataFrame(data=X, columns=ds_cancer.feature_names)
df_cancer = df_cancer.assign(CLASS=y_names)
df_cancer.iloc[100:110, 0::3]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean area</th>
      <th>mean concavity</th>
      <th>mean fractal dimension</th>
      <th>perimeter error</th>
      <th>compactness error</th>
      <th>symmetry error</th>
      <th>worst texture</th>
      <th>worst smoothness</th>
      <th>worst concave points</th>
      <th>CLASS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100</th>
      <td>13.610</td>
      <td>582.7</td>
      <td>0.08625</td>
      <td>0.05871</td>
      <td>2.8610</td>
      <td>0.014880</td>
      <td>0.01465</td>
      <td>35.27</td>
      <td>0.1265</td>
      <td>0.11840</td>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>101</th>
      <td>6.981</td>
      <td>143.5</td>
      <td>0.00000</td>
      <td>0.07818</td>
      <td>1.5530</td>
      <td>0.010840</td>
      <td>0.02659</td>
      <td>19.54</td>
      <td>0.1584</td>
      <td>0.00000</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>102</th>
      <td>12.180</td>
      <td>458.7</td>
      <td>0.02383</td>
      <td>0.05677</td>
      <td>1.1830</td>
      <td>0.006098</td>
      <td>0.01447</td>
      <td>32.84</td>
      <td>0.1123</td>
      <td>0.07431</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>103</th>
      <td>9.876</td>
      <td>298.3</td>
      <td>0.06154</td>
      <td>0.06322</td>
      <td>1.5280</td>
      <td>0.021960</td>
      <td>0.01609</td>
      <td>26.83</td>
      <td>0.1559</td>
      <td>0.09749</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>104</th>
      <td>10.490</td>
      <td>336.1</td>
      <td>0.02995</td>
      <td>0.06481</td>
      <td>2.3020</td>
      <td>0.022190</td>
      <td>0.02710</td>
      <td>23.31</td>
      <td>0.1219</td>
      <td>0.03203</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>105</th>
      <td>13.110</td>
      <td>530.2</td>
      <td>0.20710</td>
      <td>0.07692</td>
      <td>2.4100</td>
      <td>0.029120</td>
      <td>0.01547</td>
      <td>22.40</td>
      <td>0.1862</td>
      <td>0.19860</td>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>106</th>
      <td>11.640</td>
      <td>412.5</td>
      <td>0.07070</td>
      <td>0.06520</td>
      <td>2.1550</td>
      <td>0.023100</td>
      <td>0.01565</td>
      <td>29.26</td>
      <td>0.1688</td>
      <td>0.12180</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>107</th>
      <td>12.360</td>
      <td>466.7</td>
      <td>0.02643</td>
      <td>0.06066</td>
      <td>0.8484</td>
      <td>0.010470</td>
      <td>0.01251</td>
      <td>27.49</td>
      <td>0.1184</td>
      <td>0.08442</td>
      <td>BENIGN</td>
    </tr>
    <tr>
      <th>108</th>
      <td>22.270</td>
      <td>1509.0</td>
      <td>0.42640</td>
      <td>0.07039</td>
      <td>10.0500</td>
      <td>0.086680</td>
      <td>0.03112</td>
      <td>28.01</td>
      <td>0.1701</td>
      <td>0.29100</td>
      <td>MALIGNANT</td>
    </tr>
    <tr>
      <th>109</th>
      <td>11.340</td>
      <td>396.5</td>
      <td>0.05133</td>
      <td>0.06529</td>
      <td>1.5970</td>
      <td>0.015570</td>
      <td>0.01568</td>
      <td>29.15</td>
      <td>0.1699</td>
      <td>0.08278</td>
      <td>BENIGN</td>
    </tr>
  </tbody>
</table>
</div>

We'll produce a basic split into train and test sets.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f'train: X={X_train.shape} y={y_train.shape}')
print(f'test : X={X_test.shape} y={y_test.shape}')
```

    train: X=(398, 30) y=(398,)
    test : X=(171, 30) y=(171,)

And finally we'll standardize the features.
```python

# Note: each feature is standardized individually:
mu_X = np.mean(X_train, axis=0) # (N, D) -> (D,)
sigma_X = np.std(X_train, axis=0)

# Note: Broadcasting (N, D) with (D,) -> (N, D)
X_train_sc = (X_train - mu_X) / sigma_X 

# Note: Test set must be transformed identically to training set
X_test_sc = (X_test - mu_X) / sigma_X

print(f'{mu_X.shape=}, {sigma_X.shape=}')
```

    mu_X.shape=(30,), sigma_X.shape=(30,)

### Model Implementation

We can now implement the model based on the above definitions. To make it cleaner, we'll implement it as a class with an API that conforms to the `sklearn` models. See `sklearn`'s [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression) class.

```python
class BinaryLogisticRegression(object):
    def __init__(self, n_iter=100, learn_rate=0.1):
        self.n_iter = n_iter
        self.learn_rate = learn_rate
        self._w = None
        
    def _add_bias(self, X: np.ndarray):
        # Add a bias term column
        ones_col = np.ones((X.shape[0], 1))
        return np.hstack([ones_col, X])
    
    def predict_proba(self, X: np.ndarray, add_bias=True):
        X = self._add_bias(X) if add_bias else X
        
        # Apply logistic model
        z = np.dot(X, self.weights) # (N, D) * (D,)
        return logistic(z) # shape (N,)
    
    def predict(self, X: np.ndarray):
        proba = self.predict_proba(X)
        
        # Apply naive threshold of .5
        return np.array(proba > .5, dtype=np.int)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape # X is (N, D), y is (N,)
        
        # Initialize weights
        self._w = np.random.randn(d + 1) * 0.1
        
        Xb = self._add_bias(X)

        # Training loop
        self._losses = []
        for i in range(self.n_iter):
            # Predicted probabilities
            y_hat = self.predict_proba(Xb, add_bias=False)
            
            # Pointwise loss
            loss = -y.dot(np.log(y_hat)) - ((1 - y).dot(np.log(1 - y_hat)))
            
            # See gradient derivation above
            loss_grad = 1/n * Xb.T.dot(y_hat - y)  # dl/dw: (D+1, N) * (N,)
            
            # Optimization step
            self._w += -self.learn_rate * loss_grad
            self._losses.append(loss)
            
        return self
    
    @property
    def weights(self):
        if self._w is None:
            raise ValueError("Model is not fitted")
        return self._w
```

Notice that we implemented the training loop in the `fit()` method. We can now use it to train the model and show a basic loss curve.
```python
lr_model = BinaryLogisticRegression(n_iter=500, learn_rate=0.01)
lr_model.fit(X_train_sc, y_train)

plt.plot(lr_model._losses, label='$L_{\mathcal{S}}$');
plt.xlabel('training iteration');
plt.legend(); plt.grid(True);
plt.title('Train loss');
```

![png|darkmodeinvert](dl-tut2-output_56_0.png)

On this toy dataset, our performance is quite good. This is just a useful sanity check that we implemented the model correctly.
```python
y_train_pred = lr_model.predict(X_train_sc)
train_acc = np.mean(y_train_pred == y_train)
print(f'train set accuracy: {train_acc*100:.2f}%')

y_test_pred = lr_model.predict(X_test_sc)
test_acc = np.mean(y_test_pred == y_test)
print(f'test set accuracy: {test_acc*100:.2f}%')
```

    train set accuracy: 96.73%
    test set accuracy: 95.91%

## Part 2: Multiclass Logistic Regression with pytorch

What if we actually have $C$ classes? Can we still use logistic regression?

A naïve approach: train $C$ binary logistic regression classifiers, for example in a One vs. Rest scheme, and then predict based on the classifier returning the greatest probability.

One major drawback of this approach is that it doesn't model a probability distribution over the possible classes, $P_{\vec{Y}|\vec{X}}$. For example, a sample might be classified as class A with probability $0.8$ and class $B$ with $0.7$ since nothing constrains the different classifiers. Also, without *calibrating* each model, their raw scores cannot reliably be compared even though they're in the same range.

Let's introduce a better approach, which extends logistic regression to the multi-class setting.

### The softmax function

Softmax is a function which generates a discrete probability distribution for our $C$ classes given raw prediction scores. It's defined as follows:
$$
\mathrm{softmax}(\vec{z}) = \frac{e^{\vec{z}}}{\sum_{j=1}^{C} e^{z_j}}
$$
note that this is a vector-valued and multivariate function. The exponent in the enumerator operates elementwise on its vector input. The result of softmax is a vector with elements in $[0,1]$ that all sum to $1$.

### The multiclass model

Our model can now be defined as
$$
\hat{\vec{y}} = h(\vec{x}) = \mathrm{softmax}(\mattr{W}\vec{x}+\vec{b})
$$
where,
* $\hat{\vec{y}}$ is a $C\times 1$ vector of class probabilities.
* ${\vec{x}}$ is a $D\times 1$ sample.
* $\mat{W}$ is a $D\times C$ matrix representing the per-class weights.
* $\vec{b}$ is a per-class bias vector of length $C$.

Probabilistic interpretation: $\hat{y}_j = P(\rvar{Y}=j|\rvec{X}=\vec{x})$, i.e. we model the discrete probability distribution over the possible classes.

While not very powerful on it's own, this type of model is commonly found at the end of deep neural networks performing classification tasks.

The target variable is usually specified simply as an index of the correct class.
However, to train this model we need our labels to also be discrete probability distributions instead of simply a label.

We'll transform our labels to a 1-hot encoded vector corresponding to a singleton distribution (all mass is on a single class). For example, if $y^i=j$ then we'll create
$$
\vec{y}^i = [0,\dots,0,\underbrace{1}_{j\mathrm{th\ component}},0,\dots,0]^\top,
$$
and this will be the target variable corresponding to $\vec{x}^{i}$ for training.

### Cross-Entropy loss

After defining the 1-hot label vectors, the multiclass extension of the binary cross-entropy is straightforward:

$$
\ell(\vec{y}, \hat{\vec{y}}) = - \vectr{y} \log(\hat{\vec{y}})
$$

Note that only the probability assigned to the correct class affects the loss, because $\vec{y}$ here is zero in all entries except for one.

Minimizing this cross entropy can be interpreted as trying to move the probability distribution of model predictions towards the singleton distribution of the appropriate class.

### Dataset

This time we'll tackle an image classification task, the MNIST database of handwritten digits. These days this is also considered a toy dataset, even though it was widely used in the past to benchmark classification algorithms.

```python
import os
import torch
import torch.autograd
import torch.utils.data
import torchvision
import torchvision.transforms
import plot_utils

from torch import Tensor
```

We'll load the data using standard `pytorch` datasets and data loaders. We'll also need to define the transforms that should be applied to each image in the dataset before returning it.
```python
tf_ds = torchvision.transforms.ToTensor()

batch_size = 64
data_root = os.path.expanduser("~/.pytorch-datasets")

# Training and test datasets
ds_train, ds_test = [
    torchvision.datasets.MNIST(root=data_root, download=True, train=train, transform=tf_ds)   
    for train in [True, False]
]

# Data loaders
dl_train = torch.utils.data.DataLoader(ds_train, batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))

x0, y0 = ds_train[0]
n_features = torch.numel(x0)
n_classes = 10
```

Let's see what the first few samples look like.
```python
print(f'x0: {x0.shape}, y0: {y0}')
plot_utils.dataset_first_n(ds_train, 10, cmap='gray');
```

    x0: torch.Size([1, 28, 28]), y0: 5
    
![png|darkmodeinvert](dl-tut2-output_71_1.png)

Note that when training, we're actually working with **batches** of samples, as we'll be using stochastic gradient descent (SGD). So each input image is actually a four-dimensional tensor:

```python
x0, y0 = next(iter(dl_train))
x0.shape
```

    torch.Size([64, 1, 28, 28])

### Model Implementation

This time we'll use `pytorch` tensors and its [`autograd`](https://pytorch.org/docs/stable/autograd.html) functionality to implement our model. This means we wont have to implement any gradient calculations!

First, let's implement $\mathrm{softmax}(\cdot)$. We need a small numerical trick to prevent large numbers from exploding the exponentiation. You can verify that this doesn't influence the result.

```python
def softmax(z: Tensor) -> Tensor:
    """
    softmax(z)= e^(z) / sum(e^z)
    :param z: A batch of C class scores per N samples, shape (N, C).
    :returns: softmax per sample, of shape (N, C).
    """

    # normalization trick to prevent numerical instability:
    # shift so that the highest class score (per sample) is 0
    zmax, _ = torch.max(z, dim=1, keepdim=True)
    z = z - zmax # note broadcasting: (N,C) - (N,1)
    
    exp_z = torch.exp(z) # (N, C)
    sum_exp = torch.sum(exp_z, dim=1, keepdim=True) # (N, 1)
    return exp_z / sum_exp # probabilities, (N,C)
```

Let's test our softmax and calculate its derivative with `autograd`.
```python
z = torch.randn(size=(4,3), requires_grad=True)
y = softmax(z)
y
```

    tensor([[0.1167, 0.7073, 0.1760],
            [0.0702, 0.7838, 0.1460],
            [0.0755, 0.4197, 0.5048],
            [0.0205, 0.1090, 0.8704]], grad_fn=<DivBackward0>)

To calculate gradient $dL/dz$, we'll can use `autograd.grad()` as follows:
```python
y = softmax(z)
L = torch.sum(y) # scalar function of z 

torch.autograd.grad(L, z)
```

    (tensor([[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]),)

Instead of calling `autograd.grad()` directly with specific input tensors, `pytorch` provides us with a way to calculate the derivative of a tensor w.r.t. all the tensors which are **leaves** in it's computation graph (only $\vec{z}$ in this case).

This can be done by calling `.backward()` on a scalar tensor. As a result, the `.grad` property of leaf tensors will be populated with the gradient:

```python
# Example with two leaves in the computaion graph
z1 = torch.randn(size=(4,3), requires_grad=True)
z2 = torch.randn(size=(1,3), requires_grad=True)
z = z1 - z2

y = softmax(z)
L = torch.sum(y) # scalar function of z 
L.backward()     # Calculate derivative w.r.t. all leaves

z1.grad, z2.grad # The leaves z1, z2 will have their .grad populated
```

    (tensor([[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]),
     tensor([[0., 0., 0.]]))

We can visualize the resulting computation graph, and ensure that it corresponds to how we implemented our model.

```python
import torchviz
torchviz.make_dot(L, params=dict(z1=z1, z2=z2))
```

![svg|400](dl-tut2-output_84_0.svg)

The next ingredient of the solution is the cross-entropy loss function.

```python
def cross_entropy_loss(y: Tensor, y_hat: Tensor, eps=1e-6):
    """
    :param y:  Onehot-encoded ground-truth labels, shape (N, C)
    :param y_hat: A batch of probabilities, shape (N,C)
    :returns: Cross entropy between y and y_hat.
    """
    return torch.sum( - y * torch.log(y_hat + eps) )
```

Recall that we need to encode our ground-truth labels as one-hot vectors to apply the multiclass cross-entropy. We can implement this functionality as a small utility function:
```python
def onehot(y: Tensor, n_classes: int) -> Tensor:
    """
    Encodes y of shape (N,) containing class labels in the range [0,C-1] as one-hot of shape (N,C).
    """
    y = y.reshape(-1, 1) # Reshape y to (N,1)
    zeros = torch.zeros(size=(len(y), n_classes), dtype=torch.float32) # (N,C)
    ones = torch.ones_like(y, dtype=torch.float32)
    
    # scatter: put items from 'src' into 'dest' at indices correspondnig to 'index' along 'dim'
    y_onehot = torch.scatter(zeros, dim=1, index=y, src=ones)
    
    return y_onehot # result has shape (N, C)
```

If we apply it to a vector of class labels, we can see that each label gets expanded to a tensor where only the corresponding index is set to 1.
```python
onehot(torch.tensor([1, 3, 5, 0]), n_classes=10)
```

    tensor([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

Our model itself will just be a class which holds the parameters tensors $\mat{W}$ and $\vec{b}$, and applies them to an input batch $\mat{X}$. Note that applying the model is a implemented in the `forward()` function. Note also that this implementation does not use `pytorch`'s `Module` class. We'll instead keep it as simple as possible, and only use`pytroch` for its tensors and automatic differentiation.

```python
class MCLogisticRegression(object):
    def __init__(self, n_features: int, n_classes: int):
        # Define our parameter tensors: notice that now W and b are separate
        # Specify we want to track their gradients with autograd
        self.W = torch.randn(n_features, n_classes, requires_grad=True)
        self.b = torch.randn(n_classes, requires_grad=True)
        self.params = [self.W, self.b]
    
    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, X: Tensor):
        """
        :param X: A batch of samples, (N, D)
        :return: A batch of class probabilities, (N, C)
        """
        # X is (N, D), W is (D, C), b is (C,)
        z = torch.mm(X, self.W) + self.b
        y_hat = softmax(z)
        return y_hat # (N, C)
```

Let's try out the model and loss on the first batch. Note that we naïvely treat each pixel as a separate feature. We'll learn how to properly work with images in a future post.

```python
model = MCLogisticRegression(n_features, n_classes)

# Flatten images and convert labels to onehot
x0_flat = x0.reshape(-1, n_features)
y0_onehot =  onehot(y0, n_classes)

print(f'x0_flat: {x0_flat.shape}')
print(f'y0_onehot: {y0_onehot.shape}\n')
```

    x0_flat: torch.Size([64, 784])
    y0_onehot: torch.Size([64, 10])
    

We can also run a forward pass and compute loss:
```python
y0_hat = model(x0_flat)
loss = cross_entropy_loss(y0_onehot, y0_hat)
print('loss = ', loss)

# Backward pass to populate .grad on leaf nodes
loss.backward()
```

    loss =  tensor(611.5060, grad_fn=<SumBackward0>)

Since we specified `require_grad=True` for our model parameters, every operation performed on these tensors is recorded, and a **computation graph** can be built, which included the model and loss calculation.

Notice that the **leaves** in this graph are our parameters $\mat{W}$ and $\vec{b}$.

```python
import torchviz
torchviz.make_dot(loss, params=dict(W=model.W, b=model.b))
```

![svg|400](dl-tut2-output_95_0.svg)

This graph is what allows efficient implementation of the **back-propagation** algorithm which you'll learn about in the next lecture.

By calling `.backward()` from the final loss tensor, pytorch automatically populated the `.grad` property of all leaves in this graph, without us having to explicitly specify them (`W` and `b`).

### Training

The optimization will be as before, but now we'll take the gradients from the `grad` property of our parameter tensors. Therefore, the optimizer needs access **only** to the parameter tensors from the model. In fact,`pytorch`'s `Optimizer` classes work in the same way.

As before, we'll implement this from scratch using only pytorch tensors and no other build-in features.

```python
from typing import Sequence

class SGDOptimizer:
    """
    A simple gradient descent optimizer.
    """
    def __init__(self, params: Sequence[Tensor], learn_rate: float):
        self._params = params
        self._learn_rate = learn_rate
    
    def step(self):
        """
        Updates parameters in-place based on their gradients.
        """
        with torch.autograd.no_grad(): # Don't track this operation
            for param in self._params:
                if param.grad is not None:
                    param -= self._learn_rate * param.grad
    
    def zero_grad(self):
        """
        Zeros the parameters' gradients if they exist.
        """
        for param in self._params:
            if param.grad is not None:
                param.grad.zero_()
```

Without training anything yet, we can calculate the prediction accuracy as a sanity check.

```python
def evaluate_accuracy(dataloader, model, max_batches=None):
    n_correct = 0.
    n_total = 0.
    for i, (X, y) in enumerate(dataloader):
        X = X.reshape(-1, n_features) # flatten images into vectors
        
        # Forward pass
        with torch.autograd.no_grad():
            y_hat = model(X)
        
        predictions = torch.argmax(y_hat, dim=1)
        n_correct += torch.sum(predictions == y).type(torch.float32)
        n_total += X.shape[0]
        
        if max_batches and i+1 >= max_batches:
            break
        
    return (n_correct / n_total).item()

test_set_acc = evaluate_accuracy(dl_test, MCLogisticRegression(n_features, n_classes))
print(f'Test set accuracy pre-training: {test_set_acc*100:.2f}%')
```

    Test set accuracy pre-training: 7.65%

We can see that without training, the models accuracy is roughly $1/C$, i.e. it's no better than a random guess.

### The training loop

This is the crucial part of any ML pipeline where model parameters get updated iteratively.

When using `pytorch`, the training loop will generally contain the following steps:

* Each epoch:
    * Split training data into batches
    * For each batch
        * Forward pass: Compute predictions and build computation graph
        * Calculate loss
        * Set existing gradients to zero
        * Backward pass: Use back-propagation algorithm to calculate the gradients
        * Optimization step: Use the gradients to update the parameters
    * Evaluate accuracy on validation set

Notice that one pass over the entire training data is called an **epoch**.

Let's define some sane training hyper-parameters and instantiate the model.
```python
epochs = 10
max_batches = 50  # limit batches so training is fast (just as a demo)
learn_rate = .005
num_samples = len(ds_train)

# Instantiate the model we'll train
model = MCLogisticRegression(n_features, n_classes)

# Instantiate the optimizer with model's parameters
optimizer = SGDOptimizer(model.params, learn_rate=learn_rate)
```

Now we can start training. Below is the implementation of the training loop based on what we outlined above.

```python
# Epoch: traverse all samples
for e in range(epochs):
    cumulative_loss = 0

    # Loop over randdom batches of training data
    for i, (X, y) in enumerate(dl_train):
        
        X = X.reshape(-1, n_features)
        y_onehot = onehot(y, n_classes)
        
        # Forward pass: predictions and loss
        y_hat = model(X)
        loss = cross_entropy_loss(y_onehot, y_hat)
        
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Backward pass: calculate gradients 
        loss.backward() 
        
        # Update model using the calculated gradients
        optimizer.step()
        
        cumulative_loss += loss.item()
        if i+1 > max_batches:
            break

    # Evaluation
    test_accuracy = evaluate_accuracy(dl_test, model, max_batches)
    train_accuracy = evaluate_accuracy(dl_train, model, max_batches)
    
    avg_loss = cumulative_loss/num_samples
    print(f"Epoch {e}. Avg Loss: {avg_loss:.3f}, Train Acc: {train_accuracy*100:.2f}, Test Acc: {test_accuracy*100:.2f}")
```

    Epoch 0. Avg Loss: 0.352, Train Acc: 41.69, Test Acc: 43.67
    Epoch 1. Avg Loss: 0.161, Train Acc: 63.34, Test Acc: 63.29
    Epoch 2. Avg Loss: 0.099, Train Acc: 70.88, Test Acc: 71.77
    Epoch 3. Avg Loss: 0.075, Train Acc: 74.84, Test Acc: 76.31
    Epoch 4. Avg Loss: 0.064, Train Acc: 78.12, Test Acc: 78.12
    Epoch 5. Avg Loss: 0.058, Train Acc: 79.03, Test Acc: 79.77
    Epoch 6. Avg Loss: 0.053, Train Acc: 80.03, Test Acc: 81.39
    Epoch 7. Avg Loss: 0.050, Train Acc: 80.84, Test Acc: 82.08
    Epoch 8. Avg Loss: 0.051, Train Acc: 81.88, Test Acc: 83.58
    Epoch 9. Avg Loss: 0.047, Train Acc: 83.00, Test Acc: 83.77

### Final notes

* This is a very naive implementation, for example because
    * We didn't treat the images properly.
    * We didn't include any regularization.

* PyTorch provides many functions and classes that we could have used, for example:
  * Fully connected layer with model parameters
  * Softmax
  * SGD and many other optimizers
  * Cross entropy loss

    However, the purpose here was to show an (almost) from-scratch implementation using only tensors, in order to see whats "under the hood" (more or less) of the PyTorch functions.
