# Course 1

## Course introduction

- General concept of data science: Extracting *knowledge* and
  *insights* from data. There is a huge variety of methods to do it,
  ranging from basic statistics to deep neural networks.
- Machine learning systems learn how to combine *input* to produce
  useful *predictions* on *never-before-seen* data.
- General concepts of machine learning:
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
    a more succinct representations of the data.
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

## Practical work

### Environment setup

- [Anaconda](https://www.anaconda.com/distribution/): Anaconda is a
  Python distribution with great machine learning integration working
  on Linux, Windows and Mac OS X. Using it instead of the default
  Python distribution of the OS will allow us to have a finer control
  without the need for administrator privileges.
- [Setting up a virtual
      environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
      A virtual environment is a tool that helps to keep dependencies
      required by different projects separate by creating isolated
      python *virtual* environments for them. It also provides a way
      to easily reproduce the environment required to run a specific
      piece of code. This is one of the most important tools that most
      of the developers use, even outside of the data science
      world. In this course we will use Anaconda virtual environments.

### Numpy

Numpy practical work, mostly from [Python Data Science Handbook](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.00-Introduction-to-NumPy.ipynb)

#### Creation of NumPy arrays

We can use `np.array` to create arrays from Python lists:

```python
>>> import numpy as np # Standard way to import NumPy
>>> a = np.array([1, 2, 3, 4, 5])
>>> a
array([1, 2, 3, 4, 5])
```

All the values in a NumPy array have to contain the same type or be
compatible via up-cast:

```python
>>> np.array([3.14, 2, 3, 4])
array([ 3.14, 2. , 3. , 4. ])
```

NumPy arrays can be multi-dimensional. Here we create a 2 dimensional
array using a list of lists.

```python
>>> a = np.array([range(i, i + 3) for i in [2, 4]])
>>> a
array([[2, 3, 4],
       [4, 5, 6]])

>>> a.ndim
2

>>> a.shape
(2, 3)

>>> a.size
6

>>> a.dtype
dtype('int64')
```

`a.ndim` tells us that we have created a 2D array. `a.shape` tells us
that the first dimension being of length 2 (rows) and the second one
being of length 3 (columns). `a.size` tells that the array contains a
total of `6` values. `a.dtype` tells us that the array contains
integers. It is important to note that these are not Python arbitrary
precision integers. In order to perform fast computations on a large
quantity of data NumPy (which is written in C) uses internally low
level types (more explanations on this subject
[here](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb)).

There are many efficient ways to create a variety of NumPy arrays, you
can take a look at some of them
[here](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.01-Understanding-Data-Types.ipynb#Creating-Arrays-from-Scratch).

Exercises, from [ML+](https://www.machinelearningplus.com/python/101-numpy-exercises-python/):
- Create a 1D array of numbers from 0 to 9

```python
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

- Create a 3×3 NumPy array of all True’s

```python
array([[ True,  True,  True],
       [ True,  True,  True],
       [ True,  True,  True]])
```

#### Indexing

Now that we can create NumPy arrays, let's learn how to access
the values they contain.

For 1D NumPy arrays, indexing elements can be done similarly to Python
lists:

```python
>>> a = np.array([5, 0, 3, 3, 7, 9])
>>> a
array([5, 0, 3, 3, 7, 9])

>>> a[0]
5

>>> a[2]
3

>>> a[-1]
9

>>> a[-2]
7
```

In a multi-dimensional array, items can be accessed using a
comma-separated tuple of indices:

```python
>>> a = np.array([[3, 5, 2, 4], [7, 6, 8, 8], [1, 6, 7, 7]])

>>> a
array([[3, 5, 2, 4],
       [7, 6, 8, 8],
       [1, 6, 7, 7]])

>>> a[0]
array([3, 5, 2, 4])

>>> a[0][2]
2

>>> a[0, 2]
2

>>> a[-1, -2]
7

>>> a[-1, -2] = 10
>>> a
array([[ 3,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6, 10,  7]])
```

#### Slicing

Like with usual Python lists, we can create subarrays using the
*slice* notation. The NumPy slicing follows that of the standard
Python list; to access a slice of an array `x`, use this:

```python
x[start:stop:step]
```

If any of these are unspecified, they default to the values `start =
0`, `stop = size of dimension`, `step = 1`.

```python
>>> x = np.arange(10)
>>> x
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> x[:5] # first five elements
array([0, 1, 2, 3, 4])

>>> x[5:] # elements after index 5
array([5, 6, 7, 8, 9])

>>> x[4:7] # middle sub-array
array([4, 5, 6])

>>> x[::2] # every other element
array([0, 2, 4, 6, 8])

>>> x[1::2] # every other element, starting at index 1
array([1, 3, 5, 7, 9])
```

When `step` value is negative, the defaults for `start` and `stop` are
swapped. This becomes a convenient way to reverse an array:

```python
>>> x[::-1]
array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

>>> x[5::-2]
array([5, 3, 1])
```

We can also slice multidimensional arrays. Like with indexing, we will
specify the slicing indices for each dimension separated by `,`.

```python
>>> x = np.array([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])

>>> x[:2, :3] # first two rows, first three columns
array([[12,  5,  2],
       [ 7,  6,  8]])

>>> x[:, ::2] # all rows (: is a slicing with default start and stop), every other column
array([[12,  2],
       [ 7,  8],
       [ 1,  7]])
```

An _extremely important_ concept to keep in mind when working with
NumPy slices is that they return *views* rather than *copies* of the
array data. When slicing Python lists, the slice is a copy of the
original array, this is not the case with `np.array` slices.

```python
>>> x = np.array([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])

>>> x2 = x[:2, :2]
>>> x2
array([[12,  5],
       [ 7,  6]])

>>> x2[0, 0] = 99
>>> x2
array([[99,  5],
       [ 7,  6]])

>>> x
array([[99,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

Keeping this fact in mind will spare you from terrible debugging
sessions.

Exercises:
All the exercises use the following array:

```python
>>> x = np.array([[12,  5,  2,  4], [ 7,  6,  8,  8], [ 1,  6,  7,  7]])
>>> x
array([[12,  5,  2,  4],
       [ 7,  6,  8,  8],
       [ 1,  6,  7,  7]])
```

1. Select the last line of `x`

```
array([1, 6, 7, 7])
```

2. Slice the two sevens on the last line of `x`

```python
array([7, 7])
```

3. (_harder_) Slice and reverse the lines and the columns of the top right rectangle

```python
array([[8, 8, 6],
       [4, 2, 5]])
```

#### Shape manipulation

Another useful type of operation is reshaping of arrays. The most
flexible way of doing this is with the `reshape` method. For example,
if you want to put the number `1` through `9` in a `3x3` grid, you can
do the following:

```python
>>> np.arange(1, 10).reshape(3, 3)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

You can use `-1` as a joker when reshaping, NumPy will deduce the
correct value from the number of elements of the array.

```python
>>> np.arange(1, 10).reshape(3, -1)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
```

A common thing to do when working with arrays is to flatten them using
`.reshape(-1)`.

```python
>>> x = np.arange(1, 10).reshape(3, -1)
>>> x
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>>> x.reshape(-1)
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
```

Another common reshaping pattern is the conversion of a
one-dimensional array into a two-dimensional row or column
matrix. This can be done with the reshape method, or more easily by
making use of the `np.newaxis` keyword within a slice operation:

```python
>>> x = np.array([1, 2, 3])
>>> x.reshape((1, 3)) # row vector via reshape
array([[1, 2, 3]])

>>> x[np.newaxis, :] # row vector via newaxis
array([[1, 2, 3]])

>>> x.reshape((3, 1)) # column vector via reshape
array([[1],
       [2],
       [3]])

>>> x[:, np.newaxis] # column vector via newaxis
array([[1],
       [2],
       [3]])
```

Exercises:
1. Create a 3D array containing the numbers from `1` to `27` with shape
  `(3, 3, 3)`

```python
array([[[ 1,  2,  3],
      [ 4,  5,  6],
      [ 7,  8,  9]],

     [[10, 11, 12],
      [13, 14, 15],
      [16, 17, 18]],

     [[19, 20, 21],
      [22, 23, 24],
      [25, 26, 27]]])
```

2. Create the following array

```python
array([[ 0,  2],
       [ 4,  6],
       [ 8, 10],
       [12, 14],
       [16, 18]])
```

3. Using the answer to question 2, create the following array

```python
array([[16, 18],
       [12, 14],
       [ 8, 10],
       [ 4,  6],
       [ 0,  2]])
```

4. Using the answer to question 2, create the following array

```python
array([[18, 16],
       [14, 12],
       [10,  8],
       [ 6,  4],
       [ 2,  0]])
```

5. (_harder_) Create the following array

```python
array([ 2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11])
```

#### Combination

It is possible to combine multiple arrays into one, and to conversely
split a single array into multiple arrays.

First, to concatenate multiple 1D arrays, we can simply do the following:

```python
>>> x = np.array([1, 2, 3])
>>> y = np.array([4, 5, 6])
>>> np.concatenate([x, y])
array([1, 2, 3, 4, 5, 6])
>>> np.concatenate([x, y, x])
array([1, 2, 3, 4, 5, 6, 1, 2, 3])
```

We can also concatenate multidimensional arrays by precising the axis
(dimension) along which we want to perform the concatenation:

```python
>>> x = np.arange(6).reshape(2, 3)
>>> x
array([[0, 1, 2],
       [3, 4, 5]])

>>> y = np.arange(6, 12).reshape(2, 3)
>>> y
array([[ 6,  7,  8],
       [ 9, 10, 11]])

>>> np.concatenate([x, y], axis = 0) # Concatenate along dimension 0 (rows)
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

>>> np.concatenate([x, y]) # The default concatenation axis is 0
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])

>>> np.concatenate([x, y], axis = 1) # Concatenate along dimension 1 (columns)
array([[ 0,  1,  2,  6,  7,  8],
       [ 3,  4,  5,  9, 10, 11]])
```

When working with array of mixed dimensions, it can be clearer to use
[np.vstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html)
and
[np.hstack](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack).

Exercises:
All the exercises use the following arrays:

```python
>>> x = np.array([1, 2, 3])
>>> y = np.arange(6).reshape(2, 3)
>>> y
array([[0, 1, 2],
       [3, 4, 5]])
```

1. Concatenate `x` and `y` to create the following array. Be careful
   of the shapes of the array you are manipulating.

```python
array([[1, 2, 3],
       [0, 1, 2],
       [3, 4, 5]])
```

2. (_harder_) Using `x`, `y` and `np.concatenate` create the following array

```python
array([[0, 1, 2, 1, 2, 3],
       [3, 4, 5, 1, 2, 3]])
```

#### Aggregations

Multiple aggregations methods are available in NumPy. Here are a few examples:

```python
>>> x = np.arange(10)
>>> x
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> x.sum()
45

>>> x.mean()
4.5

>>> x.std() # Standard deviation
2.8722813232690143

>>> x.min()
0

>>> x.max()
9
```

Similarly to the `np.concatenate` function, we can precise along which
axis we want to perform the computation.

```python
>>> x = np.arange(12).reshape(3, 4)
>>> x
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> x.sum(axis = 0) # Sum along the rows, we have one result by column
array([12, 15, 18, 21])

>>> x.sum(axis = 1) # Sum along the columns, we have one result by row
array([ 6, 22, 38])

>>> x.mean(axis = 0) # Mean along the rows, we have one result by column
array([4., 5., 6., 7.])

>>> x.mean(axis = 1) # Mean along the columns, we have one result by row
array([1.5, 5.5, 9.5])

>>> x.min(axis = 0) # Minimum along the rows, we have one result by column
array([0, 1, 2, 3])

>>> x.min(axis = 1) # Minimum long the columns, we have one result by row
array([0, 4, 8])
```

Exercise:
1. Compute, for each column, the mean of the values of the rows of even index in
   the following array.

```python
>>> x = np.arange(42).reshape(6, 7)
>>> x
array([[ 0,  1,  2,  3,  4,  5,  6],
       [ 7,  8,  9, 10, 11, 12, 13],
       [14, 15, 16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25, 26, 27],
       [28, 29, 30, 31, 32, 33, 34],
       [35, 36, 37, 38, 39, 40, 41]])
```

The result should be

```python
array([14., 15., 16., 17., 18., 19., 20.])
```

#### Broadcasting

Now that we know basic manipulations on NumPy arrays, we can start
making operations between them.

When arrays have identical sizes, binary operations are performed on
an element-by-element basis:

```python
>>> a = np.array([1, 2, 3])
>>> b = np.array([5, 5, 5])
>>> a + b
array([6, 7, 8])
```

*Broadcasting* allows these types of binary operations to be performed
on arrays of different sizes. For example, we can just add a scalar
(think of it as a zero-dimensional array) to an array:

```python
>>> a + 5
array([6, 7, 8])
```

You can think of this as an operation that stretches or duplicates the
value `5` into the array `[5, 5, 5]`, and adds the results. The
advantage of NumPy's broadcasting is that this duplication of values
does not actually take place, but it is a useful mental model as we
think about broadcasting.

```python
>>> a
array([1, 2, 3])

>>> M = np.ones((3, 3))
>>> M
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])

>>> M + a
array([[2., 3., 4.],
       [2., 3., 4.],
       [2., 3., 4.]])
```

Here the one-dimensional array `a` is stretched, or *broadcast* across
the dimension in order to match the shape of `M`.

Broadcasting is a powerful tool that follows precise rules. An
in-depth explanation of these rules can be found
[here](https://nbviewer.jupyter.org/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/02.05-Computation-on-arrays-broadcasting.ipynb#Rules-of-Broadcasting).

A typical use case of broadcasting is to normalize the values of an
array by subtracting the mean and dividing by the standard
deviation. In the following example, let's assume that the values of
each column are observation of a different variable. We would like to
compute the mean and standard deviation for each column and then
perform the normalization.

```python
>>> x = np.random.randn(3, 5) # Randomly generate x using a normal distribution
>>> x
array([[-0.59859761,  1.3957146 ,  0.77132016,  0.80628127,  1.23405505],
       [-0.20710744,  0.13401764,  0.60230327,  0.47255149,  0.94740413],
       [-0.52074332,  0.84169282, -2.00490693,  0.82329957,  0.25557519]])

>>> x.mean(axis = 0) # Compute the mean for each column
array([-0.44214946,  0.79047502, -0.21042783,  0.70071078,  0.81234479])

>>> x.std(axis = 0) # Compute the standard deviation for each column
array([0.16921167, 0.51635727, 1.27076305, 0.16148251, 0.41072008])

>>> (x - x.mean(axis = 0)) / x.std(axis = 0) # Standardize each column with its own
                                             # mean and standard deviation
array([[-0.9245707 ,  1.17213335,  0.77256574,  0.65375807,  1.02675831],
       [ 1.38904139, -1.27132397,  0.63956149, -1.41290403,  0.3288355 ],
       [-0.46447069,  0.09919062, -1.41212722,  0.75914596, -1.35559381]])
```

Exercises:
All the exercises use the following array:

```python
>>> a = np.arange(10).reshape(5, 2)
>>> a
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7],
       [8, 9]])
```

1. Add `2` to the first column and `-1` to the second column of `a`

```python
array([[ 2,  0],
       [ 4,  2],
       [ 6,  4],
       [ 8,  6],
       [10,  8]])
```

2. Multiply the values of the first column of `a` by the maximum of
   the values of the second column. The result should be:

```python
array([[ 0,  1],
       [18,  3],
       [36,  5],
       [54,  7],
       [72,  9]])
```

#### Comparisons, Masks and Boolean logic

NumPy also provides binary comparison operators that output boolean
arrays:

```python
>>> a = np.array([1, 2, 3, 4])
>>> b = np.array([3, 2, 1, 4])
>>> a == b
array([False,  True, False,  True])

>>> a != b
array([ True, False,  True, False])

>>> a > b
array([False, False,  True, False])

>>> a <= b
array([ True,  True, False,  True])
```

These operators support the broadcasting:

```python
>>> a = np.array([1, 2, 3, 4])
>>> a == 2
array([False,  True, False, False])

>>> a % 2 == 0
array([False,  True, False,  True])
```

We can also combine boolean arrays using boolean operators

```python
>>> a = np.arange(1, 11)
>>> a
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

>>> a % 2 == 0
array([False,  True, False,  True, False,  True, False,  True, False, True])

>>> a % 5 == 0
array([False, False, False, False,  True, False, False, False, False, True])

>>> ~(a % 5 == 0) # Unary not
array([ True,  True,  True,  True, False,  True,  True,  True,  True, False])

>>> (a % 2 == 0) | (a % 5 == 0) # Binary or
array([False,  True, False,  True,  True,  True, False,  True, False, True])

>>> (a % 2 == 0) & (a % 5 == 0) # Binary and
array([False, False, False, False, False, False, False, False, False, True])
```

A very powerful NumPy feature is the ability to select elements of an
array using *boolean masks*. The mask should have the same shape (or
compatible) as the array.

```python
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> a % 3 == 0 # Multiples of 3 mask
array([ True, False, False,  True, False, False,  True, False, False, True])

>>> a[a % 3 == 0] # Select the multiples of 3 using the mask
array([0, 3, 6, 9])

>>> a % 2 == 0 # Multiples of 2 mask
array([ True, False,  True, False,  True, False,  True, False,  True, False])

>>> ~(a % 2 == 0) # Numbers that are not multiples of 2
array([False,  True, False,  True, False,  True, False,  True, False, True])

>>> a[(a % 3 == 0) & ~(a % 2 == 0)] # Select the elements that are multiple
                                    # of 3 but not of 2
array([3, 9])
```

Boolean masking also works in multiple dimensions:

```python
>>> a = np.arange(12).reshape(3, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

>>> a >= 2
array([[False, False,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True,  True,  True]])

>>> a[a >= 2]
array([ 2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

>>> a < 10
array([[ True,  True,  True,  True],
       [ True,  True,  True,  True],
       [ True,  True, False, False]])

>>> a[a < 10]
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> a[(a >= 2) & (a < 10)]
array([2, 3, 4, 5, 6, 7, 8, 9])
```

Exercises:
All the exercises use the following arrays:

```python
>>> a = np.arange(16).reshape(4, 4)
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
```

1. Select and sum the multiples of 3 in `a`, the result should be 45.

2. Subtract the mean of the values strictly greater than 8 not
   multiple of 4 to the whole array. The result should be:

```python
array([[-12., -11., -10.,  -9.],
       [ -8.,  -7.,  -6.,  -5.],
       [ -4.,  -3.,  -2.,  -1.],
       [  0.,   1.,   2.,   3.]])
```
