import argparse
import os
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1234)

# command line arguments
parser = argparse.ArgumentParser(description='Train a model with PyTorch.')
parser.add_argument('inxfile', type=str, help='Input file containing the x training data')
parser.add_argument('inyfile', type=str, help='Input file containing the y training data')
parser.add_argument('outdir', type=str, help='Output directory for the trained model')
args = parser.parse_args()

# read in the pre-processed X, y data
cols = ['f1', 'f2', 'f3', 'f4', 'l1', 'l2', 'l3']
X = pd.read_csv(args.inxfile, names=cols[0:-3])
y = pd.read_csv(args.inyfile, names=cols[-3:])

# model parameters
input_size = 4
num_classes = 3
hidden_size = 5
learning_rate = 0.1
num_epoch = 10000

# define model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)

# choose optimizer and loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epoch):
    X_tensor = Variable(torch.from_numpy(X.as_matrix()).float())
    Y_tensor = Variable(torch.from_numpy(y.as_matrix()).float())

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X_tensor)
    loss = criterion(out, Y_tensor)
    loss.backward()
    optimizer.step()

# export the model
torch.save(net.state_dict(), os.path.join(args.outdir, 'model.pt'))
