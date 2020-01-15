# Deep Natural Language Processing

## Lesson 1

### Course

#### General presentation of Data Science

General concept of data science: Extracting *knowledge* and *insights*
from data. There is a huge variety of methods to do it, ranging from
basic statistics to deep neural networks.

Machine learning systems learn how to combine *input* to produce
useful *predictions* on *never-before-seen* data.

General concepts of machine learning:
  - Labels: A *label* is the thing we're predicting - the `y` variable
    in simple linear regression (_y = ax + b_). The label could be the
    future price of wheat, the kind of animal shown in a picture, the
    meaning of an audio clip, or just about anything.
  - Features: A *feature* is an input variable - the `x` variable in a
    simple linear regression (_y = ax + b_). A simple machine learning
    project might use a single feature, while a more sophisticated
    machine learning project could use millions of features, specified
    as: _x<sub>1</sub>_,_x<sub>2</sub>_, ..., _x<sub>n</sub>_.
  - Models: A *model* defines the relationship between features and
    label - `ax + b` in a simple linear regression (_y = ax + b_). For
    example, a spam detection model might associate certain features
    strongly with "spam". Let's highlight two phases of a model's
    life. *Training* means creating or *learning* the model. That is,
    you show the model labeled examples and enable the model to
    gradually learn the relationship between features and label. In
    the case of a linear regression, it consists in finding "good"
    values for `a` and `b`, the *parameters* of the model. *Inference*
    means applying the trained model to unlabeled examples. That is,
    you use the trained model to make useful predictions (`y'`). For
    example, during inference, you can predict for a new incoming mail
    whether it is spam or not.
  - Supervised learning: In *supervised* learning, we model the
    relationship input and output. Depending on the type of
    values predicted by the model, we call it either a
    *regression* model or a *classification* model. A regression
    model predicts continuous values. For example, regression
    models make predictions that answer questions like "What is
    the value of a house in California?" or "What is the
    probability that a user will click on this ad?". A
    classification model predicts discrete values. For example,
    classification models make predictions that answer questions
    like "Is a given message spam or not spam?" or "Is this an
    image of a dog, a cat, or a hamster?".
  - Unsupervised learning: In *unsupervised* learning, we model
    the features of a dataset without reference to any label. It
    is often described as "letting the dataset speak for
    itself". These type of machine learning model include tasks
    such as *clustering* or *dimensionality
    reduction*. Clustering algorithms identify distinct groups
    of data while dimensionality reduction algorithms search for
    a more succint representations of the data.
  - There exist more learning paradigms such as [Reinforcement
    learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
    (AlphaGo), [Semi-supervised
    learning](https://en.wikipedia.org/wiki/Semi-supervised_learning)
    (hybrid between supervised and unsupervised) and
    [Self-supervised
    learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html)
    (a kind of unsupervised learning in which we use the data
    itself to create a "virtual" supervision, an example of
    self-supervised model would be a learned
    compressor-decompressor).

#### Deep learning

Deep learning is a part of machine learning methods based on
*artificial neural networks*. Artificial neural networks are a family
of computation models _vaguely_ inspired by the biological neural
networks that constitute animal brains. We will go briefly over the
history of this discipline and then study the example of the linear
regression. While not a neural network per se, understanding all its
concepts should give a very solid base for studying deep neural
networks.

##### History

A short history of deep learning (from
[Wikipedia](https://en.wikipedia.org/wiki/Timeline_of_machine_learning)):
- 1951: First neural network machine by Marvin Minsky
- 1957: Invention of the
  [Perceptron](https://en.wikipedia.org/wiki/Perceptron) by Frank
  Rosenblatt
- 1970: [Automatic
  differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
  by Seppo Linnainmaa
- 1980: [Neocognitron](https://en.wikipedia.org/wiki/Neocognitron),
  the inspiration for [convolutional neural
  networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
  by Kunihiko Fukushima
- 1982: [Hopfield
  networks](https://en.wikipedia.org/wiki/Hopfield_network), a type of
  recurrent neural network
- 1997: [Long short-term
  memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
  recurrent neural networks by Sepp Hochreiter and Jürgen Schmidhuber
- 1998: Neural networks on [MNIST](http://yann.lecun.com/exdb/mnist/),
  an digit image classification dataset

##### Linear regression

Inspired by [ML crash
course](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)

Finding "good" parameters `a` and `b` of a line `y = ax + b` to
approximate a set of point is an example of machine learning. The set
of points `(x, y)` is the *dataset* or *training data* of the
problem. In this case, the *model* is a Linear Regression (a line of
equation `y = ax + b`).  Finding "good" parameters for `a` and `b`
using labelled examples is the *training process*, *optimization* or
*learning* of the model and to define what being a "good model" means
we will need a *loss function*.

#### Loss function

The *loss function* used to evaluate a model. It penalizes bad
prediction. If the model's prediction is perfect, the loss is zero
otherwise, the loss is greater. The goal of *training* model is to
find a set of parameters (in our case `a` and `b`) that have low loss,
on average, across all examples.

![Two linear regressions](figures/linear_regression_loss.png)

Notice that the arrows in the left plot are much longer than their
counterparts in the right plot. Clearly, the line in the right plot is
a much better predictive model than the line in the left plot.

You might be wondering whether you could create a mathematical
function - a loss function - that would aggregate the individual
losses in a meaningful fashing.

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

![MSE loss](figures/mse_loss.gif)

where:
- `(x, y)` is an example in which
    - `x` is the set of features (for example, chirps / minute, age,
      gender) that the model uses to make predictions.
    - `y` is the example's label (for example, temperature).
- `prediction(x)` is a function of the weights and bias in combination
  with the set of features `x`. In our case case `prediction(x) = ax +
  b`.
- `D` is a data set containing many labeled examples, which are `(x,
  y)` pairs.
- `N` is the number of examples in `D`.

Altough MSE is commonly-used in machine learning, it is neither the
only practical loss function nor the best loss function for all
circumstances. Because of the squaring operation, a single *large*
difference between a prediction and a label will be penalized more
than many smaller ones.

A *high loss value* signifies that the models' predictions are poor
approximation of the labels. Conversely a *small loss value* means
that our model captures the structure of the data. Now that we know
that, we will take a look at algorithms designed to lower the loss of
a specific model on a specific dataset by modifying its parameters.

#### Training algorithms

##### Iterative approach

The procedure that we will use to learn our model is *iterative*. We
start with a random guess for each parameter of our model (here `a`
and `b`), we compute the loss to evaluate how good our current
parameters are and, using this loss, we will compute an update hoping
to lower the loss on the next iteration.

The following figure summarizes the process:

![Gradient Descent Diagram](figures/gradient_descent_diagram.svg)

To be able to apply this procedure to our problem, we have to find a
way to compute parameters updates.

##### Gradient descent

Suppose we had the time and the computing resources to calculate the
loss for all possible values of `a` (w<sub>1</sub> in the
figures). For the kind of regression problems we have been examining,
the resulting plot of loss vs `a` will always be *convex*. In other
words, the plot will always be bowl-shaped, kind of like this:

![Gradient Descent explanation 1](figures/gd_01.svg)

Convex problems have only one minimum; that is, only one place where
the *slope* is exactly 0. The minimum is where the loss function
converges.

As computing the loss function for all values of `a` to find its
minimum would be extremely ineficient we need a better
mechanism. Let's examine such an algorithm, the *gradient descent*.

As explained in the previous section, we start with a random guess for
our parameter.

![Gradient Descent explanation 2](figures/gd_02.svg)

Now that we have an initial value for our parameter `a`, we can
compute the loss value for our linear regression. Next, we would like
to know whether we should *increase* or *decrease* the value of `a` to
make the loss decrease.

To get this information, the gradient descent algorithm calculates the
*gradient* of the loss curve at the current point. In the next Figure,
the gradient of the loss is equal to the derivative (slope) of the
curve and tells you that you should *increase* the value of `a` to
make the loss value decrease.

![Gradient Descent explanation 3](figures/gd_03.svg)

As the gradient of a curve approximates it well in a very small
neighborhood, we add a small fraction (this fraction is called
*learning rate*) of the gradient magnitude to the starting point.

![Gradient Descent explanation 4](figures/gd_04.svg)

Now that we have a new (and hopefully better) value for `a`, we
iterate the previous process to improve it even further.

##### Learning rate

The *learning rate* is an *hyperparameter* in this problem, a
*parameter of the training process*, not of the model. It is important
to note that the learning rate (which determines the size of the steps
during the descent process) should neither be too small (otherwise the
training process will take very long to converge) nor too big (which
can lead to the process not converging at all).

[Interactive graph of the learning rate
importance](https://developers.google.com/machine-learning/crash-course/fitter/graph)

##### Gradient descent on many parameters

In the example that we have seen earlier we used the gradient descent
to find the correct value for a single parameter. In cases where we
have more parameters (`a` and `b` for a line equation), we compute the
gradient of the loss function (the derivatives according to each of
the variable) and update them all at once.

Let's compute the gradients of `a` and `b` in our example. First lets
recall and develop the MSE loss formula in our case.

![MSE formula raw](figures/mse_lin_reg.gif)

Now we want to differentiate this function according to `a` and `b`
separately. Let's start with `a`. In order to simplify the
computation we factor the previous formula by `a`.

![MSE formula a factored](figures/mse_lin_reg_loss_a.gif)

from there we easily compute the derivative of the loss according to
`a`.

![MSE formula a gradient](figures/mse_lin_reg_grad_a.gif)

Now onto `b`, we follow a similar process. First we factor by `b`.

![MSE formula b factored](figures/mse_lin_reg_loss_b.gif)

from there we compute the derivate of `l` according to `b`.

![MSE formula b gradient](figures/mse_lin_reg_grad_b.gif)

An animation of an application of the gradient descent algorithm on a
linear regression problem can be found [here](figures/anim_gd.pdf).

##### Stochastic version

In real problems, it is often not practical to compute the value of
the average loss across all of the training set as it often contains a
huge quantity of data.

To deal with this problem, we run the gradient descent using
*batches*. A batch is a subset of the training set (usually 10 to 1000
examples) that we use to compute an approximation of the gradient. The
larger the size of the batches the better the gradient approximation,
the more robust the training procedure is. By using this algorithm, we
trade robustness for efficiency.

##### Takeaways

The main takeaways of this section are the following:
- Artificial neural network have existed for a long time.
- The two main components that made them "work" were the
  backpropagation algorithm (automatic differentiation) and the advent
  of more powerful computers.
- Artificial Neural Networks are huge derivable functions that we
  optimize using a variant of the Stochastic gradient descent
  algorithm.

### Practical work

PyTorch tensor practical work, inspired from [Python Data Science
Handbook](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.00-Introduction-to-NumPy.ipynb)

In this practical work, we will setup a Python environment and
discover PyTorch tensors. Tensors are the PyTorch version of NumPy
[ndarray](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html)
with additional capabilities specific to GPU use and automatic
differentiation. Most the methods available on NumPy arrays have an
equivalent with PyTorch tensors but the name may be different.

#### Anaconda setup

[Anaconda](https://www.anaconda.com/distribution/): Anaconda is a
Python distribution with great machine learning integration working on
Linux, Windows and Mac OS X. Using it instead of the default Python
distribution of the OS will allow us to have a finer control without
the need for admnistrator privileges.

During this course, we will a *virtual environments*. A virtual
environment is a tool that helps to keep dependencies required by
different projects separate by creating isolated Python *virtual*
environments for them. It also provides a way to easily reproduce the
environment required to run a specific piece of code. This is one of
the most important tools that most of the developers use, even outside
of the data science world. In this course we will use Anaconda virtual
environments instead of `virtualenv` or `pipenv` because Anaconda
environments are able to keep track of packages installed with
`conda` (Anaconda package manager).

Downloading the Python 3.7 version of Anaconda from [this
address](https://www.anaconda.com/distribution/)

Launch the installation shell script.

```shell
> chmod u+x Anaconda3-2019.10-Linux-x86_64.sh
> ./Anaconda3-2019.10-Linux-x86_64.sh
```

The installer will ask you if it should modify your `.bashrc` to add
the anaconda folders to the `PATH`, answer yes.

Now that anaconda in installed, we will create a virtual environment
in which we will setup all our libraries. The documentation for the
anaconda virtual environment management is available
[here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

I have already created a virtual environment containing all the
libraries that we will use during this course. Download the export
file [here](env_files/deep_nlp.yml) and import the environment using
the following command:

```shell
conda env create -f deep_nlp.yml
```

#### Creation of PyTorch tensors

We can use `torch.tensor` to create tensors (arrays) from Python
lists:

```python
>>> import torch
>>> torch.tensor([1,2,3,4,5])
tensor([1, 2, 3, 4, 5])
```

All the values in a PyTorch tensor have to contain the same type or be
compatible via up-cast:

```python
>>> torch.tensor([3.14, 2, 3, 4])
tensor([3.1400, 2.0000, 3.0000, 4.0000])
```

NumPy arrays can be multi-dimensional. Here we create a 2 dimensional
array using a list of lists.

```python
>>> a = torch.tensor([range(i, i + 3) for i in [2, 4]])
>>> a
tensor([[2, 3, 4],
        [4, 5, 6]])

>>> a.ndim
2

>>> a.shape
torch.Size([2, 3])

>>> a.dtype
torch.int64
```

`a.ndim` tells us that we have created a 2D tensor. `a.shape` tells us
that the first dimension being of length 2 (rows) and the second one
being of length 3 (columns). `a.dtype` tells us that the tensor
contains integers stored on 64 bytes. It is important to note that
these are not Python arbitrary precision integers. In order to perform
fast computations on a large quantity of data PyTorch uses internally
low level types (more explanations on this subject
[here](https://pytorch.org/docs/stable/tensors.html)).

There are many efficient ways to create a variety of PyTorch tensors,
most of them identical to the NumPy counterparts described
[here](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb#Creating-Arrays-from-Scratch). The
official documentation for PyTorch tensor creation is available
[here](https://pytorch.org/docs/stable/torch.html#creation-ops)

Exercices, from
[ML+](https://www.machinelearningplus.com/python/101-numpy-exercises-python/):
- Create a 1D array of numbers from 0 to 9

```python
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

```python
>>> torch.arange(10)
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

- Create a 3×3 NumPy array of all True’s

```python
tensor([[True, True, True],
        [True, True, True],
        [True, True, True]])
```

#### Indexing

Now that we can create NumPy arrays, let's learn how to access
the values they contain.

For 1D tensors, indexing elements can be done similarly to Python
lists:

```python
>>> a = torch.tensor([5, 0, 3, 3, 7, 9])
>>> a
tensor([5, 0, 3, 3, 7, 9])

>>> a[0]
tensor(5)

>>> a[2]
tensor(3)

>>> a[-1]
tensor(9)

>>> a[-2]
tensor(7)
```

In a multi-dimensional tensors, items can be accessed using a
comma-separated tuple of indices:

```python
>>> a = torch.tensor([[3, 5, 2, 4], [7, 6, 8, 8], [1, 6, 7, 7]])
>>> a
tensor([[3, 5, 2, 4],
        [7, 6, 8, 8],
        [1, 6, 7, 7]])

>>> a[0]
tensor([3, 5, 2, 4])

>>> a[0][2]
tensor(2)

>>> a[0, 2]
tensor(2)

>>> a[-1, -2]
tensor(7)

>>> a[-1, -2] = 10
>>> a
tensor([[ 3,  5,  2,  4],
        [ 7,  6,  8,  8],
        [ 1,  6, 10,  7]])
```

#### Slicing

Like with usual Python lists, we can create subtensors using the
*slice* notation. The PyTorch slicing follows that of the standard
Python list; to access a slice of an array `x`, use this:

```python
x[start:stop:step]
```

If any of these are unspecified, they default to the values `start =
0`, `stop = size of dimension`, `step = 1`.

```python
>>> x = torch.arange(10)
>>> x
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> x[:5] # first five elements
tensor([0, 1, 2, 3, 4])
>>> x[5:] # elements after index 5
tensor([5, 6, 7, 8, 9])
>>> x[4:7] # middle sub-array
tensor([4, 5, 6])
>>> x[::2] # every other element
tensor([0, 2, 4, 6, 8])
>>> x[1::2] # every other element, starting at index 1
tensor([1, 3, 5, 7, 9])
```

As of today, PyTorch does not support using a [negative step
size](https://github.com/pytorch/pytorch/issues/229) to flip a
tensor. The same behavior can be obtained by using `torch.flip`
although this operation creates a copy of the tensor and not a view
(this will be explained later).

```python
>>> torch.flip(x, dims = (0,))
tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
```

We can also slice multidimensional arrays. Like with indexing, we will
specify the slicing indices for each dimension separated by `,`.

```python
>>> x = torch.tensor([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
tensor([[12,  5,  2,  4],
        [ 7,  6,  8,  8],
        [ 1,  6,  7,  7]])

>>> x[:2, :3] # first two rows, first three columns
tensor([[12,  5,  2],
        [ 7,  6,  8]])

>>> x[:, ::2] # all rows (: is a slicing with default start and stop), every other column
tensor([[12,  2],
        [ 7,  8],
        [ 1,  7]])
```

An _extremely important_ concept to keep in mind when working with
tensor slices is that they return *views* rather than *copies* of the
array data. When slicing Python lists, the slice is a copy of the
original array, this is not the case with `torch.tensor` slices.

```python
>>> x = torch.tensor([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
tensor([[12,  5,  2,  4],
        [ 7,  6,  8,  8],
        [ 1,  6,  7,  7]])

>>> x2 = x[:2, :2]
>>> x2
tensor([[12,  5],
        [ 7,  6]])

>>> x2[0, 0] = 99
>>> x2
tensor([[99,  5],
        [ 7,  6]])

>>> x
tensor([[99,  5,  2,  4],
        [ 7,  6,  8,  8],
        [ 1,  6,  7,  7]])
```

Keeping this fact in mind will spare you from terrible debugging
sessions.

Exercices:
All the exercices use the following array:

```python
>>> x = torch.tensor([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
tensor([[12,  5,  2,  4],
        [ 7,  6,  8,  8],
        [ 1,  6,  7,  7]])
```
1. Select the last line of `x`

```python
tensor([1, 6, 7, 7])
```

2. Slice the two sevens on the last line of `x`

```python
tensor([7, 7])
```

3. (_harder_) Slice and reverse the lines and the columns of the top right rectangle

```python
tensor([[8, 8, 6],
        [4, 2, 5]])
```

#### Shape manipulation

Another useful type of operation is reshaping of arrays. The most
flexible way of doing this is with the `reshape` method. For example,
if you want to put the number `1` through `9` in a `3x3` grid, you can
do the following:

```python
>>> torch.arange(1, 10).view(3, 3)
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
```

You can use `-1` as a joker when reshaping, PyTorch will deduce the
correct value from the number of elements of the array.

```python
>>> torch.arange(1, 10).view(3, -1)
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
```

A common thing to do when working with tensors is to flatten them
using `.view(-1)`.

```python
>>> x = torch.arange(1, 10).view(3, -1)
>>> x
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

>>> x.view(-1)
tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

Another common reshaping pattern is the conversion of a
one-dimensional array into a two-dimensional row or column
matrix. This can be done with the view method, or more easily by
making use of `None` keyword within a slice operation:

```python
>>> x = torch.tensor([1, 2, 3])

>>> x.view(1, 3) # Row vector via view
tensor([[1, 2, 3]])

>>> x[None, :] # Row vector via None
tensor([[1, 2, 3]])

>>> x.view(3, 1) # Column vector via view
tensor([[1],
        [2],
        [3]])

>>> x[:, None] # Column vector via None
tensor([[1],
        [2],
        [3]])
```

Exercices:
1. Create a 3D tensor containing the numbers from `1` to `27` with shape
  `(3, 3, 3)`

```python
tensor([[[ 1,  2,  3],
         [ 4,  5,  6],
         [ 7,  8,  9]],

        [[10, 11, 12],
         [13, 14, 15],
         [16, 17, 18]],

        [[19, 20, 21],
         [22, 23, 24],
         [25, 26, 27]]])
```

2. Create the following tensor

```python
tensor([[ 0,  2],
        [ 4,  6],
        [ 8, 10],
        [12, 14],
        [16, 18]])
```

3. Using the answer to question 2, create the following tensor

```python
tensor([[16, 18],
        [12, 14],
        [ 8, 10],
        [ 4,  6],
        [ 0,  2]])
```

4. Using the answer to question 2, create the following tensor

```python
tensor([[18, 16],
        [14, 12],
        [10,  8],
        [ 6,  4],
        [ 2,  0]])
```

5. (_harder_) Create the following tensor
```python
tensor([ 2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11])
```

#### Combination

It is possible to combine multiple tensors into one, and to conversely
split a single tensor into multiple tensors.

First, to concatenate multiple 1D tensors, we can simply do the
following:

```python
>>> x = torch.tensor([1, 2, 3])
>>> y = torch.tensor([4, 5, 6])

>>> torch.cat([x, y])
tensor([1, 2, 3, 4, 5, 6])

>>> torch.cat([x, y, x])
tensor([1, 2, 3, 4, 5, 6, 1, 2, 3])
```

We can also concatenate multidimensional arrays by precising the axis
(dimension) along which we want to perform the concatenation:

```python
>>> y = torch.arange(6, 12).view(2, 3)
>>> x = torch.arange(6).view(2, 3)
>>> x
tensor([[0, 1, 2],
        [3, 4, 5]])

>>> y = torch.arange(6, 12).view(2, 3)
>>> y
tensor([[ 6,  7,  8],
        [ 9, 10, 11]])

>>> torch.cat([x, y], dim = 0) # Concatenate along dimension 0 (rows)
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])

>>> torch.cat([x, y]) # The default concatenation dimension is 0
tensor([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11]])

>>> torch.cat([x, y], dim = 1) # Concatenate along dimension 1 (columns)
tensor([[ 0,  1,  2,  6,  7,  8],
        [ 3,  4,  5,  9, 10, 11]])
```

Exercices:
All the exercices use the following tensors:

```python
>>> x = torch.tensor([1, 2, 3])
>>> y = torch.arange(6).view(2, 3)
>>> y
tensor([[0, 1, 2],
        [3, 4, 5]])
```

1. Concatenate `x` and `y` to create the following tensor. Be careful
   of the shapes of the array you are manipulating.

```python
tensor([[1, 2, 3],
        [0, 1, 2],
        [3, 4, 5]])
```

2. (_harder_) Using `x`, `y` and `torch.cat` create the following array

```python
tensor([[0, 1, 2, 1, 2, 3],
        [3, 4, 5, 1, 2, 3]])
```

#### Aggregations

Multiple aggregations methods are available in NumPy. Here are a few
examples:

```python
>>> x = torch.arange(10, dtype = torch.float32) # Have to precise floating type to compute mean and std
>>> x
tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

>>> x.sum()
tensor(45.)

>>> x.mean()
tensor(4.5000)

>>> x.std()
tensor(3.0277)

>>> x.min()
tensor(0.)

>>> x.max()
tensor(9.)
```

Similarly to the `torch.cat` function, we can precise along which axis
we want to perform the computation.

```python
>>> x = torch.arange(12, dtype = torch.float32).reshape(3, 4)
>>> x
tensor([[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.],
        [ 8.,  9., 10., 11.]])

>>> x.sum(dim = 0) # Sum along the rows, we have one result by column
tensor([12., 15., 18., 21.])

>>> x.sum(dim = 1) # Sum along the columns, we have one result by row
tensor([ 6., 22., 38.])

>>> x.mean(dim = 0) # Mean along the rows, we have one result by column
tensor([4., 5., 6., 7.])

>>> x.mean(dim = 1) # Mean long the columns, we have one result by row
tensor([1.5000, 5.5000, 9.5000])

>>> x.min(dim = 0) # Minimum along the rows, we get two tensors, the first one with
                   # the value of the min for each column, the second one with the
                   # index of the row in which the minimum is.
torch.return_types.min(
values=tensor([0., 1., 2., 3.]),
indices=tensor([0, 0, 0, 0]))

>>> x.min(dim = 1) # Minimum along the columns, we get two tensors, the first one with
                   # the value of the min for each row, the second one with the
                   # index of the column in which the minimum is.
torch.return_types.min(
values=tensor([0., 4., 8.]),
indices=tensor([0, 0, 0]))
```

Exercice:

1. Compute, for each column, the mean of the values of the rows of
   even index in the following tensor.

```python
>>> x = torch.arange(49, dtype = torch.float32).view(7, 7)
>>> x
tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.],
        [ 7.,  8.,  9., 10., 11., 12., 13.],
        [14., 15., 16., 17., 18., 19., 20.],
        [21., 22., 23., 24., 25., 26., 27.],
        [28., 29., 30., 31., 32., 33., 34.],
        [35., 36., 37., 38., 39., 40., 41.],
        [42., 43., 44., 45., 46., 47., 48.]])
```

The result should be

```python
tensor([21., 22., 23., 24., 25., 26., 27.])
```

#### Broadcasting

Now that we know basic manipulations on PyTorch tensors, we can start
making operations between them.

When arrays have identical sizes, binary operations are performed on
an element-by-element basis:

```python
>>> a = torch.tensor([1, 2, 3])
>>> b = torch.tensor([5, 5, 5])
>>> a + b
tensor([6, 7, 8])
```

*Broadcasting* allows these types of binary operations to be performed
on arrays of different sizes. For example, we can just add a scalar
(think of it as a zero-dimensional tensor) to a tensor:

```python
>>> a + 5
tensor([6, 7, 8])
```

You can think of this as an operation that stretches or duplicates the
value `5` into the tensor `[5, 5, 5]`, and adds the results. The
advantage of PyTorch's broadcasting is that this duplication of values
does not actually take place, but it is a useful mental model as we
think about broadcasting.

```python
>>> a
tensor([1, 2, 3])

>>> M = torch.ones((3, 3))
>>> M
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])

>>> M + a
tensor([[2., 3., 4.],
        [2., 3., 4.],
        [2., 3., 4.]])
```

Here the one-dimensional tensor `a` is stretched, or *broadcast*
across the dimension in order to match the shape of `M`.

Broadcasting is a powerful tool that follows precise rules. An
in-depth explanation of these rules can be found
[here](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.05-Computation-on-arrays-broadcasting.ipynb#Rules-of-Broadcasting).

A typical usecase of broadcasting is to normalize the values of an
array by substracting the mean and dividing by the standard
deviation. In the following example, let's assume that the values of
each column are observation of a different variable. We would like to
compute the mean and standard deviation for each column and then
perform the normalization.

```python
>>> x = torch.randn(3, 5) # Randomly generate x using a normal distribution
>>> x
tensor([[-1.6168,  0.4247,  0.6664, -0.9946, -1.7289],
        [ 0.9720, -0.7299,  0.7018,  0.7963, -1.0579],
        [ 0.0678, -0.3628,  0.5733, -0.7313,  0.2808]])

>>> x.mean(dim = 0) # Compute the mean for each column
tensor([-0.1923, -0.2227,  0.6472, -0.3099, -0.8353])

>>> x.std(dim = 0) # Compute the standard deviation for each column
tensor([1.3139, 0.5899, 0.0663, 0.9670, 1.0232])

>>> (x - x.mean(dim = 0)) / x.std(dim = 0) # Standardize each column with its own
                                           # mean and standard deviation
tensor([[-1.0842,  1.0974,  0.2900, -0.7081, -0.8733],
        [ 0.8862, -0.8599,  0.8230,  1.1440, -0.2175],
        [ 0.1980, -0.2375, -1.1129, -0.4358,  1.0909]])
```

Exercices:

All the exercices use the following array:

```python
>>> a = torch.arange(10).view(-1, 2)
>>> a
tensor([[0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8, 9]])
```

1. Add `2` to the first column and `-1` to the second column of `a`

```python
tensor([[ 2,  0],
        [ 4,  2],
        [ 6,  4],
        [ 8,  6],
        [10,  8]])
```

2. Multiply the values of the first column of `a` by the maximum of
   the values of the second column. The result should be:

```python
tensor([[ 0,  1],
        [18,  3],
        [36,  5],
        [54,  7],
        [72,  9]])
```

#### Comparisons, Masks and Boolean logic

PyTorch also provides binary comparison operators that output boolean
arrays:

```python
>>> a = torch.tensor([1, 2, 3, 4])
>>> b = torch.tensor([3, 2, 1, 4])
>>> a == b
tensor([False,  True, False,  True])

>>> a != b
tensor([ True, False,  True, False])

>>> a > b
tensor([False, False,  True, False])

>>> a <= b
tensor([ True,  True, False,  True])
```

These operators support the broadcasting:

```python
>>> a = torch.tensor([1, 2, 3, 4])
>>> a == 2
tensor([False,  True, False, False])

>>> a % 2 == 0
tensor([False,  True, False,  True])
```

We can also combine boolean arrays using boolean operators

```python
>>> a = torch.arange(1, 11)
>>> a
tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

>>> a % 2 == 0
tensor([False,  True, False,  True, False,  True, False,  True, False,  True])

>>> a % 5 == 0
tensor([False, False, False, False,  True, False, False, False, False,  True])

>>> ~(a % 5 == 0)
tensor([ True,  True,  True,  True, False,  True,  True,  True,  True, False])

>>> (a % 2 == 0) | (a % 5 == 0)
tensor([False,  True, False,  True,  True,  True, False,  True, False,  True])

>>> (a % 2 == 0) & (a % 5 == 0)
tensor([False, False, False, False, False, False, False, False, False,  True])
```

A very powerful PyTorch feature is the ability to select elements of
an array using *boolean masks*. The mask should have the same shape
(or compatible) as the tensor.

```python
>>> a = torch.arange(10)
>>> a
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> a % 3 == 0 # Multiples of 3 mask
tensor([ True, False, False,  True, False, False,  True, False, False,  True])

>>> a[a % 3 == 0] # Select the multiples of 3 using the mask
tensor([0, 3, 6, 9])

>>> a % 2 == 0 # Multiples of 2 mask
tensor([ True, False,  True, False,  True, False,  True, False,  True, False])

>>> ~(a % 2 == 0) # Numbers that are not multiples of 2
tensor([False,  True, False,  True, False,  True, False,  True, False,  True])

>>> a[(a % 3 == 0) & ~(a % 2 == 0)] # Select the elements that are multiple of 3
                                    # but not of 2
tensor([3, 9])
```

Boolean masking also works in multiple dimensions:

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])

>>> a >= 2
tensor([[False, False,  True,  True],
        [ True,  True,  True,  True],
        [ True,  True,  True,  True]])

>>> a[a >= 2]
tensor([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

>>> a < 10
tensor([[ True,  True,  True,  True],
        [ True,  True,  True,  True],
        [ True,  True, False, False]])

>>> a[a < 10]
tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> a[(a >= 2) & (a < 10)]
tensor([2, 3, 4, 5, 6, 7, 8, 9])
```

Exercices:
All the exercices use the following arrays:

```python
>>> a = torch.arange(16).reshape(4, 4)
>>> a
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
```

1. Select and sum the multiples of 3 in `a`, the result should be 45.

2. Sustract the mean of the values strictly greater than 8 not
   multiple of 4 to the whole array. The result should be:

```python
array([[-12., -11., -10.,  -9.],
       [ -8.,  -7.,  -6.,  -5.],
       [ -4.,  -3.,  -2.,  -1.],
       [  0.,   1.,   2.,   3.]])
```

## References

- [Google machine learning crash course](https://developers.google.com/machine-learning/crash-course)
- [Python Data Science Handbook](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/Index.ipynb) by Jake VanderPlas
- [PyTorch tutorials](https://pytorch.org/tutorials/)
