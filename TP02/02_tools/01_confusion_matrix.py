from sklearn import datasets
#TODO: import what you need!

import pylab as pl

# import some data to play with
iris = datasets.load_iris()
X = #TODO: data
y = #TODO: target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = #TODO: split data using train_test_split() from sklearn.cross_validation (see api documentation)

# Run classifier
classifier = #TODO: linear SVM classifier
y_pred = #TODO: prediction

# Compute confusion matrix
cm = #TODO: confusion matrix

#TODO: print confusion matrix on console

# Show confusion matrix in a separate window
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()