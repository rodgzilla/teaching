import pdb
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# In PyTorch neural networks are classes that inherit from nn.Module
class SimpleLinearNetwork(nn.Module):
    def __init__(self):
        # The first step is to call the __init__ of nn.Module to
        # initialize PyTorch module mechanisms
        super(SimpleLinearNetwork, self).__init__()
        # Then we declare the layers of our neural network as class
        # fields.
        self.input_layer  = nn.Linear(2, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        # The list of operations that associate inputs to outputs by
        # the network are specified in the forward method

        # For all layers except the last one, we apply the layer
        # computations and then apply the relu activation
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.hidden_layer(x)
        x = F.relu(x)

        # After the last layer, we compute a log_softmax activation as
        # we are performing a classification task
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim = 1)

        return x

def generate_data(n_samples):
    # We randomly generate points (x, y) with -1 <= x, y < 1
    X              = torch.rand(n_samples, 2) * 2 - 1
    # We compute the Euclidean distance between the origin (0, 0) and
    # each sample
    dist_to_origin = (X ** 2).sum(axis = 1).sqrt()
    # radius value in order to have as many samples inside than
    # outside
    radius         = math.sqrt(2 / math.pi)
    # label is 1 for samples inside the disk and 0 for samples outside
    y              = (dist_to_origin < radius).long()

    return X, y

def evaluate(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    correct_pred = 0
    total_pred   = 0
    # As we are evaluating the model and not training it, we do not
    # the gradient information. We specify to torch that it does not
    # have to memorize the computation graph in this block.
    with torch.no_grad():
        for X, y in dataloader:
            # We compute the model prediction
            y_pred        = model(X)
            # The class predicted by the model is given for each line
            # by the index of the column containing the maximum value
            y_pred_class  = y_pred.argmax(dim = 1)
            # We have a correct prediction when the class predicted by
            # the model corresponds to the sample label
            correct_pred += (y_pred_class == y).sum().item()
            total_pred   += len(y)

    return correct_pred / total_pred

def train(model, criterion, optimizer, dataset, epochs, batch_size):
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    print(f'Initial accuracy {100 * evaluate(model, dataset, batch_size):5.3f}%')
    for epoch in range(epochs):
        for X, y in dataloader:
            # We compute the model prediction
            y_pred = model(X)
            # We compute the average loss between the model
            # predictions and the labels
            loss   = criterion(y_pred, y)
            # Now that we have a loss telling us how good were the
            # model predictions on these samples, we can compute the
            # gradients of the loss according to the model weights.
            loss.backward()
            # Now that we know how we should update each weight, we
            # ask our optimizer (here the Stochastic Gradient Descent)
            # to perform the update.
            optimizer.step()
            # Now that we are done we this training step, we clean the
            # gradient that we have computed.
            optimizer.zero_grad()
        # We evaluate our model every 10 epochs
        if epoch % 10 == 0:
            print(f'{epoch:3} -> {100 * evaluate(model, dataset, batch_size):5.3f}% accuracy')

def main():
    n_samples     = 3000
    epochs        = 300
    batch_size    = 32
    learning_rate = 1e-3
    X, y          = generate_data(n_samples)
    dataset       = TensorDataset(X, y)
    model         = SimpleLinearNetwork()
    criterion     = nn.NLLLoss()
    optimizer     = optim.SGD(
        params = model.parameters(),
        lr     = learning_rate
    )
    train(model, criterion, optimizer, dataset, epochs, batch_size)

if __name__ == '__main__':
    main()
