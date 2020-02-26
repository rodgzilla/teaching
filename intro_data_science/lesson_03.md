# Course 3

## Model example: Linear Regression

Inspired by [ML crash course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

Finding "good" parameters `a` and `b` of a line `y = ax + b` to
approximate a set of point is an example of machine learning. The set
of points `(x, y)` is the *dataset* or *training data* of the
problem. In this case, the *model* is a Linear Regression (a line of
equation `y = ax + b`).  Finding "good" parameters for `a` and `b`
using labelled examples is the *training process*, *optimization* or
*learning* of the model and to define what being a "good model" means
we will need a *loss function*.

## Loss function

The *loss function* used to evaluate a model. It penalizes bad
prediction. If the model's prediction is perfect, the loss is zero
otherwise, the loss is greater. The goal of *training* model is to
find a set of parameters (in our case `a` and `b`) that have low loss,
on average, across all examples.

![Two linear regressions](../figures/linear_regression_loss.png)

Notice that the arrows in the left plot are much longer than their
counterparts in the right plot. Clearly, the line in the right plot is
a much better predictive model than the line in the left plot.

You might be wondering whether you could create a mathematical
function - a loss function - that would aggregate the individual
losses in a meaningful fashion.

The linear regression models we'll examine here use a loss function
called *squared loss* (also known as *L<sub>2</sub> loss*). The
squared loss of a model `y' = ax + b` for a single example `(x, y)` is
as follows:

- = the square of the difference between the label and the prediction
- = (observation - prediction(x))<sup>2</sup>
- = (y - y')<sup>2</sup>
- = (y - (ax + b))<sup>2</sup>
- = (y - ax - b) <sup>2</sup>

*Mean square error (MSE)* is the average squared loss per example over
the whole dataset. To calculate MSE, sum up all the squared losses for
individual examples and then divide by the number of examples:

![MSE loss](../figures/mse_loss.gif)

where:
- `(x, y)` is an example in which
    - `x` is the set of features (for example, chirps / minute, age,
      gender) that the model uses to make predictions.
    - `y` is the example's label (for example, temperature).
- `prediction(x)` is a function of the weights and bias in combination
  with the set of features `x`. In our case case `prediction(x) = ax + b`.
- `D` is a data set containing many labeled examples, which are `(x, y)` pairs.
- `N` is the number of examples in `D`.

Although MSE is commonly-used in machine learning, it is neither the
only practical loss function nor the best loss function for all
circumstances. Because of the squaring operation, a single *large*
difference between a prediction and a label will be penalized more
than many smaller ones.

A *high loss value* signifies that the models' predictions are poor
approximation of the labels. Conversely a *small loss value* means
that our model captures the structure of the data. Now that we know
that, we will take a look at algorithms designed to lower the loss of
a specific model on a specific dataset by modifying its parameters.

## Training algorithms

### Iterative approach

The procedure that we will use to learn our model is *iterative*. We
start with a random guess for each parameter of our model (here `a`
and `b`), we compute the loss to evaluate how good our current
parameters are and, using this loss, we will compute an update hoping
to lower the loss on the next iteration.

The following figure summarizes the process:

![Gradient Descent Diagram](../figures/gradient_descent_diagram.svg)

To be able to apply this procedure to our problem, we have to find a
way to compute parameters updates.

### Gradient descent

Suppose we had the time and the computing resources to calculate the
loss for all possible values of `a` (w<sub>1</sub> in the
figures). For the kind of regression problems we have been examining,
the resulting plot of loss vs `a` will always be *convex*. In other
words, the plot will always be bowl-shaped, kind of like this:

![Gradient Descent explanation 1](../figures/gd_01.svg)

Convex problems have only one minimum; that is, only one place where
the *slope* is exactly 0. The minimum is where the loss function
converges.

As computing the loss function for all values of `a` to find its
minimum would be extremely inefficient we need a better
mechanism. Let's examine such an algorithm, the *gradient descent*.

As explained in the previous section, we start with a random guess for
our parameter.

![Gradient Descent explanation 2](../figures/gd_02.svg)

Now that we have an initial value for our parameter `a`, we can
compute the loss value for our linear regression. Next, we would like
to know whether we should *increase* or *decrease* the value of `a` to
make the loss decrease.

To get this information, the gradient descent algorithm calculates the
*gradient* of the loss curve at the current point. In the next Figure,
the gradient of the loss is equal to the derivative (slope) of the
curve and tells you that you should *increase* the value of `a` to
make the loss value decrease.

![Gradient Descent explanation 3](../figures/gd_03.svg)

As the gradient of a curve approximates it well in a very small
neighborhood, we add a small fraction (this fraction is called
*learning rate*) of the gradient magnitude to the starting point.

![Gradient Descent explanation 4](../figures/gd_04.svg)

Now that we have a new (and hopefully better) value for `a`, we
iterate the previous process to improve it even further.

### Learning rate

The *learning rate* is an *hyperparameter* in this problem, a
*parameter of the training process*, not of the model. It is important
to note that the learning rate (which determines the size of the steps
during the descent process) should neither be too small (otherwise the
training process will take very long to converge) nor too big (which
can lead to the process not converging at all).

[Interactive graph of the learning rate importance](https://developers.google.com/machine-learning/crash-course/fitter/graph)

### Gradient descent on many parameters

In the example that we have seen earlier we used the gradient descent
to find the correct value for a single parameter. In cases where we
have more parameters (`a` and `b` for a line equation), we compute the
gradient of the loss function (the derivatives according to each of
the variable) and update them all at once.

Let's compute the gradients of `a` and `b` in our example. First lets
recall and develop the MSE loss formula in our case.

![MSE formula raw](../figures/mse_lin_reg.gif)

Now we want to differentiate this function according to `a` and `b`
separately. Let's start with `a`. In order to simplify the
computation we factor the previous formula by `a`.

![MSE formula a factored](../figures/mse_lin_reg_loss_a.gif)

from there we easily compute the derivative of the loss according to
`a`.

![MSE formula a gradient](../figures/mse_lin_reg_grad_a.gif)

Now onto `b`, we follow a similar process. First we factor by `b`.

![MSE formula b factored](../figures/mse_lin_reg_loss_b.gif)

from there we compute the derivate of `l` according to `b`.

![MSE formula b gradient](../figures/mse_lin_reg_grad_b.gif)

An animation of an application of the gradient descent algorithm on a
linear regression problem can be found [here](../figures/anim_gd.pdf).

### Stochastic version

In real problems, it is often not practical to compute the value of
the average loss across all of the training set as it often contains a
huge quantity of data.

To deal with this problem, we run the gradient descent using
*batches*. A batch is a subset of the training set (usually 10 to 1000
examples) that we use to compute an approximation of the gradient. The
larger the size of the batches the better the gradient approximation,
the more robust the training procedure is. By using this algorithm, we
trade robustness for efficiency.

## Linear regression in scikit-learn

The machine learning library that we are going to use in this course
is [*scikit-learn*](https://scikit-learn.org/stable/). It contains
many classical machine learning algorithms and useful utility
functions to work with data. In this section we will see how to use to
compute a simple linear regression.

First, we are going to generate a fake dataset to work on. The task
that we want to solve is to predict the grade a student will get at an
exam depending on the number of hours he studied. We begin by
generating our *inputs*, the number of hours the student spent
studying.

```python
import numpy as np

X = np.random.rand(500) * 5
print(f'First few values {X[:5]}')
print(f'Minimum hours studied {X.min()}, maximum hours studied {X.max()}')
print(f'Mean hours studied {X.mean()}')
print(f'Standard deviation of hours studied {X.std()}')
```

```
First few values [0.19118183 0.56694275 1.58207607 2.46631309 2.11416058]
Minimum hours studied 0.0035492909742873557, maximum hours studied 4.9925859313657455
Mean hours studied 2.531017537650044
Standard deviation of hours studied 1.4049403305247998
```

Now that we have our inputs, we want to generate the corresponding
*outputs*. Here we say that the relation between the number of hours
studied `x` and the exam grade `y` is given by `y = 3x + 3` to which
we add a random noise.

```python
y = ((3 * X + 3) + np.random.randn(len(X)) * 2.5).clip(0, 20)
print(f'Minimum grade {y.min()}')
print(f'Maximum grade {y.max()}')
print(f'Mean grade {y.mean()}')
print(f'Standard deviation of the grades {y.std()}')
```

```
First few values [ 5.10506587  4.48843178  5.08244973 12.72177363 12.49244945]
Minimum grade 0.0
Maximum grade 20.0
Mean grade 10.64681663733521
Standard deviation of the grades 4.824883337121375
```

Let's plot the values using
[*matplotlib*](https://matplotlib.org/). We will use this library a
lot during this course.

```python
plt.figure(figsize = (8, 8))
plt.xlabel('Number of hours studied')
plt.ylabel('Exam grade')
plt.scatter(X, y)
```

![Hours studied to exam grade graph](../figures/grade_linear_regression.png)

We can see that even if the data is quite noisy, there exist a clear
linear relationship between *inputs* and *outputs*. Visualizing data
in such a way is always a very good first in the data analysis
process.

As we *always* do while building machine learning models, we have to
separate our dataset into two distinct parts: the *training set* and
the *validation set*. The training set will contain the data we will
use to train our model. The validation set will contain data that the
model will not see during its training. We will use the validation set
to check whether our model have *memorized* its training data or if it
has understood its structure. If we have great performance on the
training data and very poor ones on the evaluation data, it is a very
strong indicator of memorization.

To compute this split, we will use the function `train_test_split` of
sklearn. We choose to separate 10% of the dataset for evaluation
purpose.

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
```

```
(450,) (50,) (450,) (50,)
```

Now that our data is preprocessed, we are going to apply a linear
regression algorithm to it. We first create our model.

```python
from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression()
```

We have just created our linear regression model, it is not trained
yet as we have not shown it any data.

A particularity of sklearn algorithm is that most of them require
their input to be of the shape `[n_samples, n_features]`. Our dataset
is not currently following this shape, `X_train` and `X_val` are 1D
arrays. We need to transform them into 2D arrays.

```python
print(X_train.shape, X_val.shape)
X_train = X_train.reshape(-1, 1)
X_val = X_val.reshape(-1, 1)
print(X_train.shape, X_val.shape)
```

```
(450,) (50,)
(450, 1) (50, 1)
```

We can now *learn* or *train* or *fit* our model to our dataset. This
will find the most likely *slope* and *intercept* of the linear model
regarding our dataset.

```python
linear_regression_model.fit(X_train, y_train)
```

That's all! The model is now trained and we can use it to predict
values.

```python
hours_studied = np.array([1, 2, 3]).reshape(-1, 1)
print(f'Input shape: {hours_studied.shape}')
y_pred = linear_regression_model.predict(hours_studied)
print(y_pred)
```

```
Input shape: (3, 1)
[ 5.90258231  8.83783841 11.77309452]
```

We can also access the *learned parameters* of the model, the slope
and the intercept. A sklearn convention is that parameters learned
during the `fit` of a model and suffixed with an underscore character.

```python
a = linear_regression_model.coef_
b = linear_regression_model.intercept_
print(a, b)
```

```
[2.9352561] 2.9673262019190787
```

Let's visualize our model using matplotlib.

```python
plt.figure(figsize = (8, 8))
plt.xlabel('Number of hours studied')
plt.ylabel('Exam grade')
plt.scatter(X_train, y_train, alpha = .7)
plt.plot(X_train, a * X_train + b, color = 'red', alpha = .7)
```

![Linear regression fit](../figures/linear_regression_fit.png)

The model seem to approximate the function quite correctly. Let's
evaluate it using the mean squared error to precisely measure its
performances.

```python
from sklearn.metrics import mean_squared_error

train_error = mean_squared_error(y_train, linear_regression_model.predict(X_train))
val_error   = mean_squared_error(y_val, linear_regression_model.predict(X_val))
print(f'Training set mean squared error: {train_error}')
print(f'Validation set mean squared error: {val_error}')
```

```
Training set mean squared error: 6.634807122272552
Validation set mean squared error: 6.1336309680345575
```

We see that the mean squared is quite similar on the training and test
set. For this particular dataset, the error is even smaller on the
validation set, that is not usually the case.
