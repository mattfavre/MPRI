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

#from sklearn import datasets


import skimage
from skimage import data, filter, io
from skimage import transform as tf
import matplotlib.pyplot as plt


# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 18
im_path = './data/ocr_data/{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

im_npArray, label_npArray = load_data('./data/ocr_data/')
#for element in im_npArray:
#    print(element)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(im_npArray, label_npArray, test_size=0.4, random_state=0)

# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 1
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

# Get trained classifier
# TODO
clf = svm.SVC(kernel='linear', C=1)
#.fit(X_train,y_train)

y_pred = clf.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Show confusion matrix in a separate window
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
pl.ylabel('True label')
pl.xlabel('Predicted label')
pl.show()

# Load sudoku image
im_path = './data/sudokus/sudoku{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)
sudoku_img = np.array(Image.open(im_path).convert('L'))

# Extract cells
cells = extract_cells(sudoku_img)

# Add data for each cell
# TODO: iterate over cells and append features to a list
feature_list = list()
for cell_i in cells:
        cell_feature = extract_features(cell_i);
        feature_list.append(cell_feature)


# Classification
# TODO: use the classifier to predict on the list of features
y_result = 0  # Cette variable possede le resutat de la classification


# Load solution to compare with, print metrics, and print confusion matrix
y_sudoku = np.loadtxt(ver_path).reshape(81)

#matched_cell = 0

for i in range(0,y_sudoku.size):
    if y_sudoku[i] == y_result : #[i]
        matched_cell += 1

# TODO: print classification report
# TODO: show confusion matrix

# Print resulting sudoku
# TODO: print the resulting sudoku grid (use reshape() function to get a 9x9 grid print!





