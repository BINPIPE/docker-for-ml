import json
import os

from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error


# #############################################################################
# Load directory paths for persisting model and metadata

MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE = os.environ["MODEL_FILE"]
METADATA_FILE = os.environ["METADATA_FILE"]
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
METADATA_PATH = os.path.join(MODEL_DIR, METADATA_FILE)

# #############################################################################
# Load and split data
print("Loading data...")
boston = datasets.load_boston()

print("Splitting data...")
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# #############################################################################
# Fit regression model
print("Fitting model...")
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
train_mse = mean_squared_error(y_train, clf.predict(X_train))
test_mse = mean_squared_error(y_test, clf.predict(X_test))
metadata = {
    "train_mean_square_error": train_mse,
    "test_mean_square_error": test_mse
}

# #############################################################################
# Serialize model and metadata
print("Serializing model to: {}".format(MODEL_PATH))
dump(clf, MODEL_PATH)

print("Serializing metadata to: {}".format(METADATA_PATH))
with open(METADATA_PATH, 'w') as outfile:  
    json.dump(metadata, outfile)
