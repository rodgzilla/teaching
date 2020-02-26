# Practical work 2

In this practical work, we will build a linear neural network that
will classify MNIST digits.

## Dataset downloading and loading

Create a new notebook and, in a new cell, download the dataset using
the following commands:

```shell
!rm mnist_*
!wget https://github.com/rodgzilla/teaching_public/blob/master/datasets/mnist_train.pt?raw=true -O mnist_train.pt
!wget https://github.com/rodgzilla/teaching_public/blob/master/datasets/mnist_test.pt?raw=true -O mnist_test.pt
```

By using `!` at the beginning of the line, we can use shell commands
in the notebook.

To load the dataset, use `torch.load`

```python
>>> X_train, y_train = torch.load('mnist_train.pt')
>>> X_train.shape, y_train.shape
(torch.Size([60000, 28, 28]), torch.Size([60000]))
```

```python
>>> X_test, y_test = torch.load('mnist_test.pt')
>>> X_test.shape, y_test.shape
(torch.Size([10000, 28, 28]), torch.Size([10000]))
```

We can see that the dataset have two distinct parts. The *training
set* `(X_train, y_train)` composed a 60000 28x28 pixels images that we
will use to train our model and the *test set* `(X_test, y_test)`
composed of 10000 images that we will use to evaluate the
*generalization capabilities* of our model.


## Visualize images

Import `matplotlib` using the following commands

```python
import matplotlib.pyplot as plt
```

Visualize a few training images using the function
[`plt.imshow`](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.imshow.html)
along with their corresponding labels.

![Training MNIST image of 5](../figures/mnist_digit_01.png)

![Test MNIST image of 9](../figures/mnist_digit_02.png)

## Data reshaping

Once the data is loaded, reshape the tensors in order to consider the
images as 1D `float` tensors.

## Data normalization

The first step of the machine learning process consists in normalizing
our data. We want to transform the images in a way that the pixel
values such that the mean is 0 and the standard deviation is 1. We
have seen this procedure in the previous practical work.

As we do not want to *cheat* we will standardize the *test set* using
the mean and the standard deviation of the *training set* as the test
set is supposed to represent new data which could not have been used
to compute the mean and the standard deviation.

## Model building

Create a neural network that takes the 1D representation of the images
and outputs confidence levels for the the 10 digit classes. The number
of layers and the number of neurons in each layer is up to you. Use
the neural network presented during the course as a base.

## Training and evaluation code

Write a training function and an evaluation function inspired by the
ones we have seen in the course.

The training function takes both the training and testing datasets as
input. At regular intervals (for example every 500 batches), perform
an evaluation of the model on just a few batches (for example 150) of
the training set and of the testing set to check that our model is
indeed learning something.

The execution of your code should display something like this:

```
[  0][    0] -> 10.208% training accuracy 10.052% testing accuracy
[  0][  500] -> 11.010% training accuracy 11.135% testing accuracy
[  1][    0] -> 15.958% training accuracy 15.937% testing accuracy
[  1][  500] -> 29.531% training accuracy 30.531% testing accuracy
[  2][    0] -> 40.302% training accuracy 40.260% testing accuracy
[  2][  500] -> 50.958% training accuracy 50.833% testing accuracy
[  3][    0] -> 54.490% training accuracy 53.885% testing accuracy
[  3][  500] -> 54.333% training accuracy 54.667% testing accuracy
[  4][    0] -> 58.396% training accuracy 59.396% testing accuracy
[  4][  500] -> 68.510% training accuracy 68.594% testing accuracy
[  5][    0] -> 73.562% training accuracy 74.042% testing accuracy
[  5][  500] -> 77.000% training accuracy 77.417% testing accuracy
[  6][    0] -> 79.167% training accuracy 79.865% testing accuracy
[  6][  500] -> 80.688% training accuracy 81.500% testing accuracy
[  7][    0] -> 82.917% training accuracy 83.042% testing accuracy
[  7][  500] -> 83.396% training accuracy 84.604% testing accuracy
[  8][    0] -> 85.458% training accuracy 85.542% testing accuracy
[  8][  500] -> 86.333% training accuracy 86.625% testing accuracy
[  9][    0] -> 86.865% training accuracy 87.094% testing accuracy
[  9][  500] -> 87.781% training accuracy 87.646% testing accuracy
[ 10][    0] -> 87.802% training accuracy 88.094% testing accuracy
[ 10][  500] -> 87.917% training accuracy 88.615% testing accuracy
[ 11][    0] -> 88.833% training accuracy 88.812% testing accuracy
[ 11][  500] -> 88.865% training accuracy 88.990% testing accuracy
[ 12][    0] -> 88.771% training accuracy 89.344% testing accuracy
[ 12][  500] -> 89.562% training accuracy 89.635% testing accuracy
[ 13][    0] -> 89.531% training accuracy 89.740% testing accuracy
[ 13][  500] -> 89.656% training accuracy 89.833% testing accuracy
[ 14][    0] -> 89.458% training accuracy 89.844% testing accuracy
[ 14][  500] -> 89.844% training accuracy 90.260% testing accuracy
[ 15][    0] -> 90.198% training accuracy 90.385% testing accuracy
[ 15][  500] -> 90.531% training accuracy 90.562% testing accuracy
[ 16][    0] -> 90.552% training accuracy 90.490% testing accuracy
[ 16][  500] -> 90.406% training accuracy 90.844% testing accuracy
[ 17][    0] -> 90.469% training accuracy 90.844% testing accuracy
[ 17][  500] -> 90.562% training accuracy 90.969% testing accuracy
[ 18][    0] -> 90.906% training accuracy 91.104% testing accuracy
[ 18][  500] -> 90.969% training accuracy 91.281% testing accuracy
[ 19][    0] -> 91.312% training accuracy 91.260% testing accuracy
[ 19][  500] -> 90.844% training accuracy 91.521% testing accuracy
```

## Visualize errors

Write a function that allows you to visualize the images on which the
model predictions were wrong. In order to visualize these images using
`plt.imshow` you will have to undo the normalization that we performed
earlier.
