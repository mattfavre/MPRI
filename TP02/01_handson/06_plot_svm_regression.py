print(__doc__)

###############################################################################
# Generate sample data
import numpy as np

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
from sklearn.svm import SVR

svr_lin = SVR(kernel='linear', C=1e3)
svr_rbf = #TODO: SVM regression with rbf kernel gamma=0.1, C=1e3
svr_poly = #TODO: SVM regression with plynomial kernel (degree 2), C=1e3
y_lin = svr_lin.fit(X, y).predict(X)
y_rbf = #TODO: fit + predict
y_poly = #TODO: fit + predict

###############################################################################
# look at the results
import pylab as pl
pl.scatter(X, y, c='k', label='data')
pl.hold('on')
pl.plot(X, y_rbf, c='g', label='RBF model')
pl.plot(X, y_lin, c='r', label='Linear model')
pl.plot(X, y_poly, c='b', label='Polynomial model')
pl.xlabel('data')
pl.ylabel('target')
pl.title('Support Vector Regression')
pl.legend()
pl.show()