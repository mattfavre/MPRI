import random
import pylab as pl
from sklearn import datasets

# Import data
iris = datasets.load_iris()
X = iris.data[iris.target != 2] #Only load two classes
y = iris.target[iris.target != 2] #Only load two classes

# Add noisy features
random.seed(0)
for i in range(80):
    idx = random.randrange(0, len(X))
    X[idx] = random.random() * 6

# Split the data into a training set and a test set
#TODO

# Run classifier
clf = #TODO
probas_ = #TODO: fit + predict proba

# Compute Precision-Recall and plot curve
precision, recall, thresholds = #TODO: precision recall curve (probas_pred=probas_[:, 1]

pl.clf()
pl.plot(recall, precision, label='Precision-Recall curve')
pl.xlabel('Recall')
pl.ylabel('Precision')
pl.ylim([0.0, 1.05])
pl.xlim([0.0, 1.0])
pl.title('Precision-Recall example')
pl.legend(loc="lower left")
pl.show()