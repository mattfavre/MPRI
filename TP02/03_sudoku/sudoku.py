import math
from classification.metrics import show_confusion_matrix, print_classification_report
from classification.svm import load_or_train
from image.cell_extraction import extract_cells, plot_extracted_cells
from PIL import Image
import numpy as np
from image.cell_extraction import *
from image.feature_extraction import *
from sklearn import cross_validation, datasets, svm
from sklearn.metrics import confusion_matrix
import pylab as pl
from classification.svm import*

#from sklearn import datasets


import skimage
from skimage import data, filter, io
from skimage import transform as tf
import matplotlib.pyplot as plt

#im_npArray, label_npArray = load_data('./data/ocr_data/')

# Split the data into a training set and a test set
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(im_npArray, label_npArray, test_size=0.4, random_state=0)


# Get trained classifier
#clf = svm.SVC(kernel='linear', C=5)

#y_pred = clf.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
#cm = confusion_matrix(y_test, y_pred)

# Show confusion matrix in a separate window
#pl.matshow(cm)
#pl.title('Confusion matrix')
#pl.colorbar()
#pl.ylabel('True label')
#pl.xlabel('Predicted label')
#pl.colors()
#pl.show()

clf = load_or_train(False)

# Load sudoku image
sudoku_nb = 20
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)
sudoku_img = np.array(Image.open(im_path).convert('L'))

# Extract cells
cells = extract_cells(sudoku_img)

# Add data for each cell
feature_list = []
for cell_i in cells:
        cell_feature = extract_features(cell_i)
        feature_list.append(cell_feature)


# Classification
result = clf.predict(feature_list)    

# Load solution to compare with, print metrics, and print confusion matrix
result_correct = np.loadtxt(ver_path, dtype='int')

# TODO: print classification report
# TODO: show confusion matrix

print_classification_report(result_correct ,result,'Rapport')

# Print resulting sudoku
print "Resulting image recongition sudoku :"
print(result.reshape((9,9)))

print "Expected sudoku :"
print(result_correct.reshape((9,9)))


