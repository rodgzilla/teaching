import pdb

import numpy as np

import torch
from torchvision import datasets

def convert_to_tensor(dataset):
    imgs        = [img for img, _ in dataset]
    y           = torch.tensor([label for _, label in dataset])
    img_tensors = [torch.tensor(np.array(img)) for img in imgs]
    X           = torch.cat(
        [img_tensor[None, ...] for img_tensor in img_tensors],
        dim = 0
    )

    return X, y

def main():
    mnist_train      = datasets.MNIST('../../datasets', train = True, download = True)
    mnist_test       = datasets.MNIST('../../datasets', train = False, download = True)
    X_train, y_train = convert_to_tensor(mnist_train)
    X_test, y_test   = convert_to_tensor(mnist_test)
    torch.save((X_train, y_train), '../../datasets/mnist_train.pt')
    torch.save((X_test, y_test), '../../datasets/mnist_test.pt')

if __name__ == '__main__':
    main()
