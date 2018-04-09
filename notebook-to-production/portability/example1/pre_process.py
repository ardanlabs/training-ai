import argparse
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# command line arguments
parser = argparse.ArgumentParser(description='Preprocess iris training data.')
parser.add_argument('infile', type=str, help='Input file containing the training set')
parser.add_argument('outdir', type=str, help='Output directory for the pre-processed data')
args = parser.parse_args()

# read in the data
cols = ['f1', 'f2', 'f3', 'f4', 'species']
data = pd.read_csv(args.infile, names=cols)

# scale the feature and encode the labels
X = data[cols[0:-1]]
X = MinMaxScaler().fit_transform(X)
y = pd.get_dummies(data['species'])

# output the features and encoded labels
Xout = pd.DataFrame(X, columns=cols[0:-1])
Xout.to_csv(os.path.join(args.outdir, 'x_train.csv'), index=False, header=False)
y.to_csv(os.path.join(args.outdir, 'y_train.csv'), index=False, header=False)
