# Practical work 5

The goal of this practical is to build an
[autoencoder](https://en.wikipedia.org/wiki/Autoencoder) for the
dataset MNIST.

![Autoencoder](../figures/autoencoder.png)

(image from Wikipedia.org)

An autoencoder is a neural network composed of two smaller models: an
encoder and a decoder. The encoder takes the input data and process it
to reduce the number of values it is stored on, it *compresses* the
data. The decoder takes the code that was produced by the encoder and
expand it to increase the number of values it is stored on, it
*decompresses* the data. You will find below a simple multilayer
perceptron autoencoder. The loss used to train such a model is the
Mean Squared Error loss that we have seen while studying the linear
regression.

You have to build most of the code by yourself using pieces from
previous practical work and the documentation. The first goal is to
write all the code required to train the following model:

```python
class MLPAutoencoder(nn.Module):
    def __init__(self, encoding_size):
        super(MLPAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, encoding_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_flat       = torch.flatten(x, start_dim = 1)
        encoded_x    = self.encoder(x_flat)
        decoded_x    = self.decoder(encoded_x)
        decoded_x_2d = decoded_x.view(x.shape[0], 1, 28, 28)

        return decoded_x_2d
```

Once this model works, the goal is create another autoencoder using
convolution layers.

## Dataset loading

Load the MNIST training and test data. You want to load the data in
such a way that the values of the pixels are in the interval `[0,
1]`. Verify that you have loaded data properly by:

- Looking at the minimum and maximum pixel values for a few images
- Displaying a few images using `plt.imshow`

## Training and evaluation code

Write the training and evaluation function. Their prototype should be
the following ones:

```python
def train(model, epochs, optimizer, criterion, device, train_loader, test_loader)

def eval(ae, criterion, device, loader, n_batch = -1)
```

The task might be different from what we have seen until now but the
training loop should look very similar to the ones of the previous
courses.

## Main function

Write the main function combining everything you wrote and the given
autoencoder architecture.

The pseudocode is the following one:

```
Create device
Create model
Put model on device
Create optimizer (Use Adam)
Create criterion (Use nn.MSELoss())
Call the training function
```

Once your function training procedure is working, you can take a look
at your results by using `plt.imshow`.

You can find below a few images you should get during the training
process. Left column is the source image, right column is the result
of the compression-decompression process. All the images in this
section and the CNN section have been obtained with an encoding size
of 10, resulting in a compression factor of 98.7% (10 / 784).

0% training

![MLP AE 0%](../figures/MLP_AE_00.png)

50% training

![MLP AE 50%](../figures/MLP_AE_01.png)

100% training

![MLP AE 100%](../figures/MLP_AE_02.png)

## Convolutional autoencoder

First, create the build blocks of our convolutional autoencoder using
this kind of design.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3,
                 stride = 1, padding = 1, act = F.relu):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
        )
        self.act = act

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x
```

I cannot stress this enough: **test systematically each of block right
after creating it**. You do not have to create a complicated model
just to test a single module, just randomly create some inputs using
`torch.randn` and check the shapes.

For example, to test `ConvBlock` that we have just defined, we can
write the following test:

```python
conv_block = ConvBlock(in_channels = 32, out_channels = 64)
img_batch  = torch.randn(16, 32, 50, 50)
output     = conv_block(img_batch)
print(output.shape)
```

```
torch.Size([16, 64, 50, 50])
```

The batch size did not change, the number of channels of the output is
`64` as we asked and the images are still `50x50` because `ConvBlock`
uses padding by default. This module works.

The blocks that you have to build are the following ones (remember the
strided convolution from the course):

```python
class ConvDownsample(nn.Module):
    '''
    A strided convolution and a relu activation, the height and width of
    its inputs should be divided by 2.
    '''
    def __init__(self, ...):
        super(ConvDownsample, self).__init__()
        pass

    def forward(self, x):
        pass
```

Using
[`nn.ConvTranspose2d`](https://pytorch.org/docs/stable/nn.html#convtranspose2d)
build the following block. Pay very close attention to the examples in
the documentation. You can also take a look at the "Transposed
convolution animations section" of
[vdumoulin](https://github.com/vdumoulin/conv_arithmetic#transposed-convolution-animations)
repository.

```python
class ConvUpsample(nn.Module):
    '''
    Opposite of a downsample, the height and width of its inputs should
    be multiplied by 2. This class takes `output_size` as parameter. This
    parameter will be given to the ConvTranspose2d layer as in the
    documentation.
    '''
    def __init__(self, output_size, ...):
        super(ConvUpsample, self).__init__()
        pass

    def forward(self, x):
        pass
```

The following block will be located at the very end of the encoder.

```python
class ArrayToEncoding(nn.Module):
    '''
    Transforms the output of the last convolution of the encoder
    into a 1D array and make it pass through a linear layer with
    out_features = encoding_size.
    '''
    def __init__(self, encoding_size, ...):
        super(ArrayToEncoding, self).__init__()
        pass

    def forward(self, x):
        pass
```

The following block will be located at the very beginning of the
decoder.

```python
class EncodingToArray(nn.Module):
    '''
    Transforms the output of the encoder (the code) into something
    that a convolution can take as input, a 2D array. This layer
    should be composed of a linear layer with the correct number
    of out_features followed by a `view` operation.
    '''
    def __init__(self, encoding_size, ...):
        super(EncodingToArray, self).__init__()
        pass

    def forward(self, x):
        pass
```

Once all the building blocks are created and **tested**, build your
convolutional autoencoder. The skeleton should look something like
this:

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, encoding_size):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(...),
            ConvBlock(..),
            ConvDownsample(...),
            ConvBlock(...),
            ConvBlock(...),
            ConvDownsample(...),
            ArrayToEncoding(encoding_size)
        )

        self.decoder = nn.Sequential(
            EncodingToArray(encoding_size),
            ConvUpsample(...),
            ConvBlock(...),
            ConvBlock(...),
            ConvUpsample(...),
            ConvBlock(..),
            ConvBlock(...) # Remember to use sigmoid as activation here
        )

    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x
```

0% training

![Conv AE 0%](../figures/CONV_AE_00.png)

The pattern that we see in this picture is called a [checkerboard
artifact](https://distill.pub/2016/deconv-checkerboard/) and it is due
to the use of transposed convolutions.

50% training

![Conv AE 50%](../figures/CONV_AE_01.png)

100% training

![Conv AE 100%](../figures/CONV_AE_02.png)
