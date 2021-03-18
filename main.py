
import numpy as np
import time
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
import matplotlib.pyplot as plt

# This script does all of the heavy lifting.  There are multiple function calls to several files in and out of this directory


START = time.time()

#=============================

# Load raw Data
D = np.load('/Users/gianlucabencomo/PycharmProjects/DDI/data.npz', allow_pickle=True)['save_list']

# Declare parameters
params = ['Structural Similarity', 'Genotypic Similarity', 'Therapeutic Similarity', 'Phenotypic Similarity']

# We want a balanced dataset
balance_length = len(D[1]['Positive DDIs'][2][params[0]])
neg_length = len(D[2]['All Negative DDIs'][2][params[0]])

save_path = "/Users/gianlucabencomo/PycharmProjects/DDI/results/DotProduct_031721.npz"


# Declaring our covariance / kernel function
#kernel = 1.0 * RBF(1.0)
#kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
#kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
kernel = DotProduct()

# Notes to be saved with the file.
notes = input('Hi User.  Would you like to input any comments for this session run?')

#=============================
# Here we are building our feature matrix (X) where we include the values of each parameter for every + or - DDI pair.
# Our output vector, y, holds the class that each DDI pair belongs to. 0 = DDI - and 1 = DDI +
#=============================

X = np.zeros((balance_length * 2, len(params)))
y = np.zeros(balance_length * 2)

for i in range(0, len(params)):
    X[:balance_length, i] = D[1]['Positive DDIs'][i + 2][params[i]]

y[:balance_length] = 1

rand_neg = np.random.choice(neg_length, balance_length, replace=False)

for i in range(0, balance_length):
    for j in range(0, len(params)):
        X[balance_length + i, j] = D[2]['All Negative DDIs'][j + 2][params[j]][rand_neg[i]]

#=============================

folds = 5

from DDI.crossValidation import Kfold_crossVal

# Partition our X and y into folds, so each fold can be trained and tested individually
trainX, trainY, testX, testY = Kfold_crossVal(X, y, F=folds)

data = list()
for k in range(folds):
    print("Running xval fold", k+1)

    # placeholder for gp classification
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(trainX[k], trainY[k])

    # use optimized gpc model to find probabilities of each DDI belonging to a class
    probs = gpc.predict_proba(testX[k])

    # The gpc object uses a link logit function to do the classification.  This means we get a plot that looks like
    # logistic regression and the cut off for each class is a probability of 0.5
    classification = gpc.predict(testX[k])

    for i in range(0, len(probs)):
        data.append((probs[i], classification[i], testY[k][i], k))


from DDI.plot.analysisFunctions import ROC

# call Receiver Operating Characteristic and receive the integration of it
auc = ROC(data)

END = time.time()

print(f'AUC for this session: {auc}')
print(f"Time taken to process this script: {(END - START) / 60 ** 2} hrs")

#=============================
# Save all of our data so we can re-visualize
#=============================
save_dict = dict()
save_dict.update({
        "data": data,
        "auc": auc,
        "duration": END - START,
        "kernel": str(kernel),
        "notes": notes,
    })

np.savez_compressed(save_path, save_dict=save_dict)

plt.show()