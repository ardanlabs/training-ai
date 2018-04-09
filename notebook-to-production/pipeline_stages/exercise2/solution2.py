import argparse
import os
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
torch.manual_seed(1234)

# command line arguments
parser = argparse.ArgumentParser(description='Inference using a saved PyTorch model.')
parser.add_argument('inmodel', type=str, help='Input file containing the saved model')
parser.add_argument('infile', type=str, help='Input file containing the input features')
parser.add_argument('outdir', type=str, help='Output directory for the inferences')
args = parser.parse_args()

# read in the input features
cols = ['f1', 'f2', 'f3', 'f4']
infer_data = pd.read_csv(args.infile, names=cols)
infer_data = MinMaxScaler().fit_transform(infer_data)

# model parameters
input_size = 4
num_classes = 3
hidden_size = 5

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

# Load the persisted model parameters
net.load_state_dict(torch.load(args.inmodel))

# Perform the inference
X = Variable(torch.from_numpy(infer_data).float())
out = net(X)
_, labels = torch.max(out.data, 1)

species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
predictions = []
for label in labels:
    predictions.append(species[label])

# save the inferences
out_data = pd.DataFrame(predictions, columns=['predictions'])
out_data.to_csv(os.path.join(args.outdir, os.path.basename(args.infile)), index=False, header=False)
