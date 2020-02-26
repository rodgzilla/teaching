# Lesson 4

## CIFAR-10 convolutional neural network

In this section, we will explain the code of the convolutional neural
network of the previous practical work.

```python
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels  = 3,
        out_channels = 32,
        kernel_size  = 3,
        padding      = 1
    )
    self.conv2 = nn.Conv2d(
        in_channels  = 32,
        out_channels = 64,
        kernel_size  = 3,
        padding      = 1
    )
    self.fc1 = nn.Linear(
        in_features  = 8 * 8 * 64,
        out_features = 128
    )
    self.fc2 = nn.Linear(
        in_features  = 128,
        out_features = 10
    )

  def forward(self, x):
    x      = self.conv1(x)
    x      = F.relu(x)
    x      = F.max_pool2d(x, 2)
    x      = self.conv2(x)
    x      = F.relu(x)
    x      = F.max_pool2d(x, 2)
    x      = x.view(x.shape[0], -1)
    x      = self.fc1(x)
    x      = F.relu(x)
    x      = self.fc2(x)
    output = F.log_softmax(x, dim=1)

    return output
```

When sizing the first linear layer that will be applied right after
the flattening step `self.fc1` we have to compute the number of
features it takes as input. The number of features is computed as
follows:

```
in_features = height of feature map * width of feature map * number of
              channel of the last convolution layer
            = 8 * 8 * 64
```

The height and width of the feature maps are both `8` because we
started with an input shape of `[3, 32, 32]` and applied two
`max_pool2d` operations: `[32, 32] -> [16, 16] -> [8, 8]`. The shapes
are simply divided by 2 at each maximum pooling operation because we
apply *padded convolutions*.

Now for the flattening operation.

```python
x = x.view(x.shape[0], -1)
```

The input of the flattening operation will be a vector of shape
`[batch_size, 64, 8, 8]` and we would like to transform it to the
shape `[batch_size, 64 * 8 * 8]`. Even if we know the batch size that
we specified during the creation of the `DataLoader`, we are not able
to use it directly there. Our training dataset contains 50000 images
and our batch size is 64, we can't divide the dataset in equal size
batches. The dataloader will create size 64 batches as long as
possible and create a last one with what is left at the end. This
means that our last batch will contain 16 images (`50000 % 64 = 16`).

Another way to perform this operation would be to use the
`torch.flatten` operator.

```python
>>> t = torch.arange(96).view(2, 3, 4, 4)
>>> t # 2 images, 3 channel, 4 * 4 pixels
tensor([[[[ 0,  1,  2,  3],
          [ 4,  5,  6,  7],
          [ 8,  9, 10, 11],
          [12, 13, 14, 15]],

         [[16, 17, 18, 19],
          [20, 21, 22, 23],
          [24, 25, 26, 27],
          [28, 29, 30, 31]],

         [[32, 33, 34, 35],
          [36, 37, 38, 39],
          [40, 41, 42, 43],
          [44, 45, 46, 47]]],


        [[[48, 49, 50, 51],
          [52, 53, 54, 55],
          [56, 57, 58, 59],
          [60, 61, 62, 63]],

         [[64, 65, 66, 67],
          [68, 69, 70, 71],
          [72, 73, 74, 75],
          [76, 77, 78, 79]],

         [[80, 81, 82, 83],
          [84, 85, 86, 87],
          [88, 89, 90, 91],
          [92, 93, 94, 95]]]])
>>> torch.flatten(t, start_dim = 1)
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
         66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
         84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]])
>>> torch.flatten(t, start_dim = 1).shape
torch.Size([2, 48])
```

Now let's take a look at how to perform the training on GPU rather
than on CPU.

The first thing we need to do is to declare a `torch.device`.

```python
device    = torch.device('cuda')
model     = ConvNet().to(device)
```

The
[`tensor.to`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to)
is used to transfer a tensor for RAM to GPU memory. Once this
operation is done, the computations will performed and their result
stored on GPU. You cannot mix CPU tensors with GPU tensors when
applying your neural network so you either have to transfer the model
back to CPU (for example after the training is completed) or transfer
its input to GPU (what we do while training and evaluating).

Let's take a look at how the training function is modified.

```python
def train(model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if batch_idx % 350 == 0:
        print('Train Epoch: {} [{:5}/{:5} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
```

This function runs the training for a single epoch, it will be called
in a loop. As we want to perform our computations on the GPU and we
already have our model on the device, we have to transfer our batch
data on it too. Just as we did with the model, we perform this
transfer by writing

```python
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
```

We are usually not able to store the whole dataset on the GPU due to
memory limitations, that's why we transfer the data batch by batch. As
we do not keep any reference to it, it will be garbage collected at
some point after we have performed our training step.

You can also notice the line at the beginning of the function

```python
model.train()
```

This
[method](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train)
is used to set the model in *training mode* (it is its default
mode). Some layers (we will see one later in this course) have a
different behavior during their training and evaluation phase, to
indicate that we are going to train the model we use this function.

Let's now take a look at the evaluation function

```python
def test(model, device, loader, prefix, n_batch = None):
  model.eval()
  loss       = 0
  correct    = 0
  total_pred = 0
  with torch.no_grad():
      for batch_idx, (data, target) in enumerate(loader):
          if n_batch is not None and batch_idx == n_batch:
            break
          data, target = data.to(device), target.to(device)
          output       = model(data)
          loss        += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
          pred         = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          correct     += pred.eq(target.view_as(pred)).sum().item()
          total_pred  += len(target)

  loss /= total_pred

  print('\n' + prefix + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      loss, correct, total_pred,
      100. * correct / total_pred))
```

Similarly to `model.train`, to specify that we are going to evaluate
the model we use `model.eval`.

As explained in the practical work, we do not want to evaluate on the
*whole* `DataLoader` every time as it takes a significant amount of
time. We limit the number of batches used to evaluate the model using
the following mechanism

```python
for batch_idx, (data, target) in enumerate(loader):
  if n_batch is not None and batch_idx == n_batch:
      break
```

If we do not specify the `n_batch` parameter to the function, the
whole `DataLoader` will be used, otherwise we stop after `n_batches`
iterations.

We use the `reduction = 'sum'` parameter to the loss to not compute
the mean of the loss during this step. We compute the sum of the loss
over all the batches and then only we divide by the number of
predictions. If we did not, the computation would have given a bigger
weight to the losses of the last smaller batch.

Let's now take a look a the data loading code.

```python
def create_loaders():
  cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  train_dataset = datasets.CIFAR10(
    root      = '../data',
    train     = True,
    download  = True,
    transform = cifar_transform
  )

  test_dataset = datasets.CIFAR10(
    root      = '../data',
    train     = False,
    download  = True,
    transform = cifar_transform
  )

  train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
  test_loader  = DataLoader(test_dataset, batch_size = 64, shuffle = True)

  return train_loader, test_loader
```

It is important to remember that if you want to test your model on new
samples after you are done with its training, you have to apply
exactly the same normalization steps you applied on your training
data.

Let's now combine everything in a main function.

```python
def main():
  (
    train_loader,
    test_loader
  )         = create_loaders()
  device    = torch.device('cuda')
  model     = ConvNet().to(device)
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  for epoch in range(1, 15):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, train_loader, 'Train', 20)
    test(model, device, test_loader, 'Test')
```

When we run it we get the following output

```python
Train Epoch: 1 [    0/50000 (0%)]	Loss: 2.300201
Train Epoch: 1 [22400/50000 (45%)]	Loss: 1.493223
Train Epoch: 1 [44800/50000 (90%)]	Loss: 1.182059

Train set: Average loss: 0.9930, Accuracy: 826/1280 (65%)
Test set: Average loss: 1.0527, Accuracy: 6293/10000 (63%)

[...]

Train Epoch: 6 [    0/50000 (0%)]	Loss: 0.426290
Train Epoch: 6 [22400/50000 (45%)]	Loss: 0.376126
Train Epoch: 6 [44800/50000 (90%)]	Loss: 0.287188

Train set: Average loss: 0.3822, Accuracy: 1111/1280 (87%)
Test set: Average loss: 0.8457, Accuracy: 7245/10000 (72%)

[...]

Train Epoch: 10 [    0/50000 (0%)]	Loss: 0.178617
Train Epoch: 10 [22400/50000 (45%)]	Loss: 0.201108
Train Epoch: 10 [44800/50000 (90%)]	Loss: 0.385630

Train set: Average loss: 0.1913, Accuracy: 1191/1280 (93%)
Test set: Average loss: 1.1805, Accuracy: 7089/10000 (71%)

[...]

Train Epoch: 14 [    0/50000 (0%)]	Loss: 0.032493
Train Epoch: 14 [22400/50000 (45%)]	Loss: 0.035444
Train Epoch: 14 [44800/50000 (90%)]	Loss: 0.162892

Train set: Average loss: 0.0820, Accuracy: 1240/1280 (97%)
Test set: Average loss: 1.6143, Accuracy: 7109/10000 (71%)
```

We notice during the logging steps a very big difference between the
metrics on the training set and on the test set. At the end of the
training procedure the average loss is 20 times bigger on the test set
and accuracy difference is 26%. We are very clearly in a case of
*overfitting*. The model has *memorized* the training samples instead
of understanding the structure of the information.

This phenomenon is a lot more common in the deep learning field than
in the machine learning field in general. To prevent this problem, we
will use yet another family of layers called *normalization layers*.

## Normalization layers: Dropout

There exist many different normalization methods. One of them,
specific to neural networks is called
[*Dropout*](https://arxiv.org/abs/1207.0580).

The algorithm itself is very simple. We chose a probability `p` and
"drop" a value (replace it with a `0`) with this probability `p`. We
then scale all the remaining values by `1 / (1 - p)`. The scaling term
is there to compensate for the fact that a proportion of neurons are
inactive. Once the training is done, we deactivate stopping to drop
values and removing the scaling of the values.

```python
>>> dropout = nn.Dropout(p = .5) # p = .5 -> 1 / (1 - p) = 2.
>>> t = torch.arange(10).float()
>>> t
tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])

>>> dropout(t)
tensor([ 0.,  0.,  0.,  0.,  8.,  0., 12.,  0.,  0., 18.])

>>> dropout(t)
tensor([ 0.,  0.,  0.,  6.,  0., 10.,  0.,  0.,  0.,  0.])

>>> dropout(t)
tensor([ 0.,  2.,  0.,  0.,  8.,  0.,  0., 14., 16.,  0.])

>>> dropout(t)
tensor([ 0.,  0.,  0.,  6.,  0., 10.,  0.,  0.,  0., 18.])

>>> dropout(t)
tensor([ 0.,  2.,  4.,  6.,  0.,  0., 12.,  0., 16., 18.])

>>> dropout(t)
tensor([ 0.,  2.,  0.,  0.,  8., 10.,  0.,  0., 16.,  0.])
```

As explained in the research article, the dropout works by
"encouraging" each neuron to learn features that are generally helpful
for producing correct answers. It prevents complex co-adaptations in
which a feature detector is only helpful in the context of several
other specific feature detectors.

Another way to view this procedure is as a very efficient way to
perform [model
ensembling](https://en.wikipedia.org/wiki/Ensemble_learning), an
algorithm which combines multiples different models by averaging their
prediction (you can see this a vote).

Let's add some dropout layers on our CIFAR-10 CNN.

```python
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(
    in_channels  = 3,
        out_channels = 32,
        kernel_size  = 3,
        padding      = 1
    )
    self.dropout1 = nn.Dropout(
        p = 0.6
    )
    self.conv2 = nn.Conv2d(
        in_channels  = 32,
        out_channels = 64,
        kernel_size  = 3,
        padding      = 1
    )
    self.dropout2 = nn.Dropout(
        p = 0.6
    )
    self.fc1 = nn.Linear(
        in_features  = 8 * 8 * 64,
        out_features = 128
    )
    self.dropout3 = nn.Dropout(
        p = .5
    )
    self.fc2 = nn.Linear(
        in_features  = 128,
        out_features = 10
    )

    def forward(self, x):
        x      = self.conv1(x)
        x      = self.dropout1(x)
        x      = F.relu(x)
        x      = F.max_pool2d(x, 2)
        x      = self.conv2(x)
        x      = self.dropout2(x)
        x      = F.relu(x)
        x      = F.max_pool2d(x, 2)
        x      = x.view(x.shape[0], -1)
        x      = self.fc1(x)
        x      = self.dropout3(x)
        x      = F.relu(x)
        x      = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
```

Let's launch it and study the output.

```python
Train Epoch: 1 [    0/50000 (0%)]	Loss: 2.330966
Train Epoch: 1 [22400/50000 (45%)]	Loss: 1.564563
Train Epoch: 1 [44800/50000 (90%)]	Loss: 1.210437

Train set: Average loss: 1.6368, Accuracy: 732/1280 (57%)
Test set: Average loss: 1.6530, Accuracy: 5616/10000 (56%)

[...]

Train Epoch: 10 [    0/50000 (0%)]	Loss: 1.047794
Train Epoch: 10 [22400/50000 (45%)]	Loss: 0.776986
Train Epoch: 10 [44800/50000 (90%)]	Loss: 0.909246

Train set: Average loss: 1.0388, Accuracy: 967/1280 (76%)
Test set: Average loss: 1.1059, Accuracy: 7028/10000 (70%)

[...]

Train Epoch: 20 [    0/50000 (0%)]	Loss: 0.949350
Train Epoch: 20 [22400/50000 (45%)]	Loss: 0.718932
Train Epoch: 20 [44800/50000 (90%)]	Loss: 0.823930

Train set: Average loss: 0.8870, Accuracy: 1024/1280 (80%)
Test set: Average loss: 0.9860, Accuracy: 7292/10000 (73%)

[...]

Train Epoch: 29 [    0/50000 (0%)]	Loss: 0.733138
Train Epoch: 29 [22400/50000 (45%)]	Loss: 0.886871
Train Epoch: 29 [44800/50000 (90%)]	Loss: 0.755015

Train set: Average loss: 0.8390, Accuracy: 1022/1280 (80%)
Test set: Average loss: 0.9645, Accuracy: 7274/10000 (73%)
```

We see that the metrics are now much closer between the training set
and the test set. We have also increased the accuracy of the model on
its training set, which tells us that we have increased its
*generalization capabilities*.

There exist many [normalization
layers](https://pytorch.org/docs/stable/nn.html#normalization-layers)
in the literature with different use cases, dropout was only one
example. Some of the most used ones are [Batch
Normalization](https://arxiv.org/abs/1502.03167)
([`nn.BatchNorm2d`](https://pytorch.org/docs/stable/nn.html#batchnorm2d))
and [Layer Normalization](https://arxiv.org/abs/1607.06450)
([`nn.LayerNorm`](https://pytorch.org/docs/stable/nn.html#layernorm)).

## Modular PyTorch code

As we have seen in the previous section, the internal code of a neural
network can get big quite fast. To deal with this problem, neural
network in PyTorch are usually created in a modular fashion. We will
rewrite the code of the previous section using this idea.

We are starting with the following class.

```python
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(
        in_channels  = 3,
        out_channels = 32,
        kernel_size  = 3,
        padding      = 1
    )
    self.dropout1 = nn.Dropout(
        p = 0.6
    )
    self.conv2 = nn.Conv2d(
        in_channels  = 32,
        out_channels = 64,
        kernel_size  = 3,
        padding      = 1
    )
    self.dropout2 = nn.Dropout(
        p = 0.6
    )
    self.fc1 = nn.Linear(
        in_features  = 8 * 8 * 64,
        out_features = 128
    )
    self.dropout3 = nn.Dropout(
        p = .5
    )
    self.fc2 = nn.Linear(
        in_features  = 128,
        out_features = 10
    )

    def forward(self, x):
        x      = self.conv1(x)
        x      = self.dropout1(x)
        x      = F.relu(x)
        x      = F.max_pool2d(x, 2)
        x      = self.conv2(x)
        x      = self.dropout2(x)
        x      = F.relu(x)
        x      = F.max_pool2d(x, 2)
        x      = x.view(x.shape[0], -1)
        x      = self.fc1(x)
        x      = self.dropout3(x)
        x      = F.relu(x)
        x      = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
```

We have clear code repetitions for the convolution part and the fully
connected part of our model. In our example, each convolution has a
corresponding dropout rate and is always followed by a max pooling
operation.

We can isolate this pattern in a new class.

```python
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding, dropout_p):
    super(ConvBlock, self).__init__()
    self.conv_layer = nn.Conv2d(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = kernel_size,
        padding      = padding
    )
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, x):
    x = self.conv_layer(x)
    x = self.dropout(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)

    return x
```

`ConvBlock` is a new neural network that simply computes a
convolution, followed by a dropout, followed by a relu activation and
a max pooling operation. It is short and easy to understand.

We can do the same thing for the linear layers.

```python
class LinearBlock(nn.Module):
  def __init__(self, in_features, out_features, dropout_p, activation):
    super(LinearBlock, self).__init__()
    self.linear_layer = nn.Linear(
        in_features  = in_features,
        out_features = out_features
    )
    self.dropout = nn.Dropout(dropout_p)
    self.activation = activation

  def forward(self, x):
    x = self.linear_layer(x)
    x = self.dropout(x)
    x = self.activation(x)

    return x
```

Each linear layer is followed by a dropout layer (the probability may
be 0 for the last one) and a activation. As the activation may vary
(`F.relu` for the first one and `F.log_softmax` for the output layer),
we take it as a parameter.

Now that we have our building blocks for the convolutional and linear
parts, let's put them together.

```python
from functools import partial

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_part = nn.Sequential(
          ConvBlock(
              in_channels  = 3,
              out_channels = 32,
              kernel_size  = 3,
              padding      = 1,
              dropout_p    = .6
          ),
          ConvBlock(
              in_channels  = 32,
              out_channels = 64,
              kernel_size  = 3,
              padding      = 1,
              dropout_p    = .6
          )
        )

        self.linear_part = nn.Sequential(
          LinearBlock(
              in_features  = 8 * 8 * 64,
              out_features = 128,
              dropout_p    = .5,
              activation   = F.relu
          ),
          LinearBlock(
              in_features  = 128,
              out_features = 10,
              dropout_p    = 0,
              activation   = partial(F.log_softmax, dim = 1)
          )
        )

    def forward(self, x):
      x = self.conv_part(x)
      x = torch.flatten(x, start_dim = 1)
      x = self.linear_part(x)

      return x
```

In this class, we declare that our neural network will be composed of
two sequences of blocks, the first ones being convolutional and the
latter ones being linear. `nn.Sequential` simply chains a list of
`nn.Module`s sequentially.

As the activation of the last layer takes an extra parameter (`dim =
1` to indicate the dimension along which we compute the softmax), we
have to wrap it to set this parameter using `partial(F.log_softmax,
dim = 1)`. It performs the same function as `lambda x:
F.log_softmax(x, dim = 1)`. It is a
[*partial*](https://en.wikipedia.org/wiki/Partial_application)
function application.

By writing our neural network code in this modular fashion, our
`forward` method gets extremely simple and clear.

```python
def forward(self, x):
  x = self.conv_part(x)
  x = torch.flatten(x, start_dim = 1)
  x = self.linear_part(x)

  return x
```

Most advanced neural network are written this way as their computation
sequences can be very complex.

## Building "real" neural networks

For now we have only built neural networks working on toy datasets in
order to understand their mechanisms.

When working on real world problems, there are two main problems we
face when training a neural network from scratch: the amount of data
required and the computation power necessary.

The amount of data required to properly train a neural network from
scratch is usually extremely big. For example, the
[ImageNet](http://www.image-net.org/) dataset contains 14M images and
requires around 300GB of storage space to work with.

The computation power is also often extremely large. For example
reproducing the training procedure of the
[Meena](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html)
chatbot created in 2020 by Google would cost around $1.4M in cloud
computing cost only.

To deal with these problems, machine learning practitioners use a
variety of methods. In this section we will go over three of the main
ones.

### Learning rate decay

Training a neural network on a huge dataset usually requires to run a
lots of epochs (usually in the hundreds). As we have seen while
studying the gradient descent algorithm, as we get close to minimum of
the function we are optimizing, the optimizer may oscillate around the
correct values of the parameters but not reach them. As explained in
[this
video](https://www.coursera.org/lecture/deep-neural-network/learning-rate-decay-hjgIA),
this behavior may be due to learning rate that is too big. The obvious
solution to this problem would be to start the training process with a
smaller learning rate from the beginning but, as we have seen,
lowering this hyperparameter can greatly increase the time required
for the weights of the model to converge to a value minimizing the
loss.

A compromise between these approaches would be to start with a "big"
learning rate value and steadily decrease it along the training
process, this is called [*Learning rate
decay*](https://en.wikipedia.org/wiki/Learning_rate#Learning_rate_schedule).

In PyTorch, these mechanisms are implemented in the
`optim.lr_scheduler` module. In this example, the learning is
multiplied by `0.1` (`gamma`) every `7` steps (`step_size`).

```python
optimizer        = optim.SGD(
    [torch.randn(1, requires_grad=True)],
    lr=1e-3
)
exp_lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size = 7,
    gamma     = 0.1
)

for epoch in range(1, 25):
    exp_lr_scheduler.step()
    print(f'Epoch {epoch:2}, lr {optimizer.param_groups[0]["lr"]}')
```

```
Epoch 1, lr 0.001
Epoch 2, lr 0.001
Epoch 3, lr 0.001
Epoch 4, lr 0.001
Epoch 5, lr 0.001
Epoch 6, lr 0.001
Epoch 7, lr 0.0001
Epoch 8, lr 0.0001
Epoch 9, lr 0.0001
Epoch 10, lr 0.0001
Epoch 11, lr 0.0001
Epoch 12, lr 0.0001
Epoch 13, lr 0.0001
Epoch 14, lr 1.0000000000000003e-05
Epoch 15, lr 1.0000000000000003e-05
Epoch 16, lr 1.0000000000000003e-05
Epoch 17, lr 1.0000000000000003e-05
Epoch 18, lr 1.0000000000000003e-05
Epoch 19, lr 1.0000000000000003e-05
Epoch 20, lr 1.0000000000000003e-05
Epoch 21, lr 1.0000000000000002e-06
Epoch 22, lr 1.0000000000000002e-06
Epoch 23, lr 1.0000000000000002e-06
Epoch 24, lr 1.0000000000000002e-06
```

The training function usually takes the scheduler in addition to the
optimizer and performs both steps on after the other.

```python
def train(model, dataset, optimizer, scheduler, ...):
   ...
   for X, y in dataloader:
       ...
       optimizer.step()
       scheduler.step()
```

To get more detailed explanations, you can take a look at [this blog
post](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)
which compares different learning rate schedules on CIFAR-10.

### Data augmentation

[*Data
augmentation*](https://en.wikipedia.org/wiki/Convolutional_neural_network#Artificial_data)
consists in artificially increasing the size of the dataset by adding
a random noise to each sample during the training process.

![Data augmentation](../figures/data_augmentation.png)

(image from
[medium.com](https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec))

For example when we perform image classification of objects, we know
that if an image contains a dog, then its horizontal flip also
contains a dog. By randomly flipping samples while training the model,
we encourage it to develop this kind of invariance. There exist many
different kinds of augmentation methods including:

- Random cropping
- Horizontal or vertical flipping
- Color jittering

Data augmentation strategies are *domain specific*. For example, when
working with textual data, a common augmentation strategy consists in
applying the following translation `initial language -> other language
-> initial language` using an external system. The idea behind this
augmentation strategy is that, if the translation system is good
enough, the *semantic information* (its meaning) of the text should be
preserved by translation while its *lexical composition* (the precise
sequence of words) should change.

Let's now write some Python code using the `torchvision` data
augmentation methods.

![Dog original picture](../figures/dog_augmentation_base.jpg)

(image from [wagwalking](https://wagwalking.com/condition/hanging-tongue-syndrome))

```python
augmentations = [
   transforms.RandomRotation(30),
   transforms.RandomHorizontalFlip(),
   transforms.ColorJitter(brightness=1., contrast = 1.),
   transforms.RandomResizedCrop(1400)
]

fig, ax = plt.subplots(3, 4, figsize = (15, 15))
for row in range(3):
  for col in range(4):
    transformation = random.choice(augmentations)
    ax[row][col].imshow(transformation(img))
    ax[row][col].axis('off')
plt.tight_layout()
```

![Data augmentation output](../figures/data_augmentation_dog.png)

These transforms are most often used when working with the rest of the
torchvision pipeline which we already used while normalizing input
images.

```python
img_transforms = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.CIFAR10(
      root      = '../data',
      train     = True,
      download  = True,
      transform = img_transform
)
```

When fetching an element from `train_dataset`, a new set of
transformation will be applied. This mechanism allows to perform data
augmentation without changing the rest of your training loop. You also
want to create a different set of transforms *without augmentations*
for your test set.

### Transfer learning

Another very popular strategy to deal with the lack of data is to use
[*Transfer
Learning*](https://en.wikipedia.org/wiki/Transfer_learning). In this
method, we use a neural network that have been trained on another
(similar enough) task as a base to train our model for our task. We
want to *transfer* its knowledge to the new task.

The idea behind this method is that the *features* (the successive
internal representations of data) the model has learned on the base
task will probably be useful for the new one. This is where the
*similar enough* criterion between tasks comes into play. Train a
classifier to discriminate between cats and dogs will probably work a
lot better than doing the same thing starting from a galaxy image
classifier.

This methodology has been used for a long time in the field of
computer vision. It is the case because we had access to very generic
models working with real world images like ImageNet. For other fields
of deep learning like NLP, the use of transfer learning has only
started producing great results recently. The last lesson of this
course will be focused on this topic.

A lot of pretrained models weights have been released over the years
for both [PyTorch](https://pytorch.org/hub/) and
[TensorFlow](https://www.tensorflow.org/hub). These models constitute
a great toolbox for deep learning practitioners to apply transfer
learning.

Let's say we want to apply transfer learning starting from a VGG16
network trained on ImageNet to an image classification problem with 5
classes.

![VGG16 network](../figures/VGG16_architecture.png)

To apply transfer learning, we usually *freeze* most the weights of
the network (making it not trainable) in order to preserve what has
been learned on the initial task.

![VGG16 frozen](../figures/VGG16_architecture_transfer_01.png)

We then replace the last linear layer by a new untrained one (with the
correct number of outputs for our task). If our task is very different
from the initial one it is also possible to unfreeze more layers
starting from the last ones.

![VGG16 last layer replaced](../figures/VGG16_architecture_transfer_02.png)

After the model is setup, we start the training process as usual. As
we have frozen most of the network, the only parameters that will be
trained will be the ones specific to our task, located at the end of
the model.
