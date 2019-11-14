
# Import relevant packages
import torch
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torchvision.datasets.mnist import MNIST
from torch import nn
from pyro.infer import SVI, TraceMeanField_ELBO
import pyro
from pyro import poutine
import pyro.optim as pyroopt
import pyro.distributions as dist
import pyro.contrib.bnn as bnn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.utils import lazy_property
import math
from torch.utils import data

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

if __name__ == '__main__':

    X,Y = load_iris(return_X_y=True)

    for i,x in enumerate(X):
        torch.save(torch.tensor(x).float(), './data/%d.pt' %i)

    n_classes=len(set(Y)) # torch.tensor(len(set(Y))).int()
    print(f'Number of classes: {n_classes}')

    partition = {'train':[str(x) for x in range(0,100)],
                 'val':[str(x) for x in range(100,125)],
                 'test':[str(x) for x in range(125,150)]}

    labels = {str(i):torch.tensor(j).to(torch.int64) for i,j in zip(range(150), Y)}

    class Dataset(data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, list_IDs, labels):
            'Initialization'
            self.labels = labels
            self.list_IDs = list_IDs

      def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

      def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample

            ID = self.list_IDs[index]

            # Load data and get label
            X = torch.load('data/' + ID + '.pt')
            y = self.labels[ID]

            return X, y


    # CUDA for PyTorch
    device = torch.device('cpu')

    # Parameters
    params = {'batch_size': 50,
              'shuffle': True,
              'num_workers': 0}

    # Generators
    training_set = Dataset(partition['train'], labels)
    train_loader = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition['val'], labels)
    validation_generator = data.DataLoader(validation_set, **params)

    test_set = Dataset(partition['test'], labels)
    test_generator = data.DataLoader(test_set, **params)

    DROP_OUT_PROP = 0.1

    class FCN(nn.Module):
        def __init__(self, n_classes=n_classes):
            super(FCN, self).__init__()
            self.fc = nn.Sequential(nn.BatchNorm1d(num_features=4),
                                    nn.Dropout(p=DROP_OUT_PROP),
                                    nn.Linear(4, 50),
                                    nn.BatchNorm1d(num_features=50),
                                    nn.Dropout(p=DROP_OUT_PROP),
                                    nn.ReLU(),
                                    nn.Linear(50, 20),
                                    nn.BatchNorm1d(num_features=20),
                                    nn.Dropout(p=DROP_OUT_PROP),
                                    nn.ReLU(),
                                    nn.Linear(20, n_classes),
                                    nn.Softmax(dim=-1))

        def forward(self, inp):
            return self.fc(inp)

    FC_NN = FCN()

    LEARNING_WEIGHT = 1e-3

    optim = torch.optim.AdamW(FC_NN.parameters(recurse=True), lr=LEARNING_WEIGHT, weight_decay=0.001, amsgrad=True)#SGD(FC_NN.parameters(recurse=True), lr=0.1, momentum=0.95)
    epochs = 500

    train_acc = []
    test_acc = []

    FC_NN = FC_NN.float()
    for i in range(epochs):
        total_loss = 0.0
        total = 0.0
        correct = 0.0
        for x, y in train_loader:
            FC_NN.zero_grad()
            pred = FC_NN.forward(x)
            loss = nnf.binary_cross_entropy(pred, nnf.one_hot(y, torch.tensor(n_classes)).float())
            total_loss += loss
            total += y.size(0)
            correct += (pred.argmax(-1) == y).sum().item()
            loss.backward()
            optim.step()

        train_acc.append(tracc)
        test_acc.append(teacc)
        print('epoch: %d | loss: %.3f | acc: %.5f' %((i+1), total_loss, correct/total*100), end='\r')
