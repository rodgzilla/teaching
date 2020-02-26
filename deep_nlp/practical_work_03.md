# Practical work 3

The goal of this practical work is to build convolutional neural
networks to complete two image classification tasks,
[MNIST](http://yann.lecun.com/exdb/mnist/), on which we worked during
the previous course, and
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), a 32x32 RGB
image classification task.

## MNIST

### Data loading

During the last practical we did the data loading an normalization
ourselves in order to get familiar with tensor manipulations. During
this course, we will do it the "right" way by using `torchvision`
which is PyTorch library specialized in image dataset manipulations.

First, let's start with the usual imports plus a few new ones for
torchvision.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
```

We then download, standardize and create loaders for the MNIST dataset
with the following code:

```python
mnist_transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root      = '../data',
    train     = True,
    download  = True,
    transform = mnist_transform
)

test_dataset = datasets.MNIST(
    root      = '../data',
    train     = False,
    download  = True,
    transform = mnist_transform
)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader  = DataLoader(test_dataset, batch_size = 64, shuffle = True)
```

[`transforms.Normalize`](https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Normalize)
performs the mean subtraction and standard deviation division that we
coded last week.

### Training loop and evaluation loop

Using code from the previous practical work, create a training and
evaluation functions. The evaluation function takes the number of
batches used for evaluation as an optional argument. If no value is
given the whole dataloader is use otherwise we stop the evaluation
after the specified number of batches.

You can use the following code as inspiration

```python
>>> s = 'who you gonna call ?'.split()
>>> s
['who', 'you', 'gonna', 'call', '?']

>>> for word_index, word in enumerate(s):
        if word_index == 3:
            break
        print(word_index, word)
0 who
1 you
2 gonna
```

### Neural network

Create a convolutional neural network using the one we created during
the last course and the convolutional layers example of this course.

The following architecture choice is reasonable, you are not
forced to used it:
- Convolution (convolution with a padding parameter in order to not
  lose a pixel band on this outside of the picture) with `32` filters
  of size `3x3` and a padding of `1`,
- ReLU activation,
- Convolution layer with `64` filters of size `3x3` and a padding of
  `1`,
- ReLU activation,
- `2x2` max pooling
- Tensor flattening in order to be able to use linear layers. You can
  either use `tensor.view` or `torch.flatten` to do this.
- Linear layer with `128` outputs
- ReLU activation
- Linear layer (output) with 10 outputs
- `log_softmax` activation

This model architecture is essentially a miniature version of the
VGG16 network.

### Main function

Write a main function that instantiate the model, the optimizer, the
loss and calls the training and evaluation methods.

### Optimizer modification

Try to change the optimizer in your code from `SGD` to
`torch.optim.Adagrad`, `torch.optim.RMSprop` or `torch.optim.Adam` and
analyze the impact.

## CIFAR-10

Repeat the whole process with the CIFAR-10 dataset. You should only
have to change a few lines from the loading code and the architecture
of your model. This dataset is also available through the
`torchvision.datasets` interface. You have to adapt the normalization
code to take into account the three channels of CIFAR-10 images (RGB).
