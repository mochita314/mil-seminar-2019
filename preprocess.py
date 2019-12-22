from __future__ import print_function

from argparse import Namespace

from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision.transforms import Compose

import optuna
optuna.logging.disable_default_handler()

from task import Task

import math

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os

import random

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12544, 128)
        self.fc2 = nn.Linear(128, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 1 - correct / len(test_loader.dataset)

def objective(trial):

    mean = trial.suggest_uniform("mean",0.1,0.9)
    std = trial.suggest_uniform("std",0.1,0.9)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    preprocess_func = Compose([transforms.ToTensor(),transforms.Normalize((mean,mean,mean),(std,std,std))])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=True, download=True,
                          transform=preprocess_func),
        batch_size=64, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data', train=False,
                          transform=preprocess_func),
        batch_size=1000, shuffle=True, **kwargs)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    EPOCH = 10
    for epoch in range(EPOCH):
        train(args, model, device, train_loader, optimizer, epoch)
        error_rate = test(args, model, device, test_loader)
        scheduler.step()

    return error_rate

def select_preprocess(args: Namespace, task: Task) -> Compose:
    ''' Select optimal preprocess
    Given task, this method returns optimal preprocess function.
    '''
    # TODO: Implement preprocess selection
    # preprocess_func = Compose([])

    TRIAL_SIZE = 50
    study = optuna.create_study()
    study.optimize(objective,n_trials=TRIAL_SIZE)

    #最適なパラメータ
    mean = study.best_params['mean']
    std = study.best_params['std']

    preprocess_func = Compose([transforms.ToTensor(),transforms.Normalize((mean,mean,mean),(std,std,std))])
    return preprocess_func

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--task', '-t', type=str, help='Task name')

    parser.add_argument('--gpu', '-g', type=int,
                        nargs='?', help='GPU ids to use')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    ############################################################
    # Instantiate task object
    task = Task(args.task)

    ############################################################

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    preprocess_func = select_preprocess(args,task)
    #print(preprocess_func)

#最適化された時の目的関数の値
#study.best_value
#全試行結果
#study.trials


