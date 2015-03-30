from sklearn import cross_validation, datasets, svm
import random

# Import data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Add noisy features
random.seed(0)
for i in range(80):
    idx_rnd = random.randrange(0, len(X)-1)
    X[idx_rnd] = random.random() * 6

clf = svm.SVC(kernel='linear')

kfolds = 10

scores = cross_validation.cross_val_score(clf, X, y, cv=kfolds)
print scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


