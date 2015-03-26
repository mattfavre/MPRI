from classification.metrics import show_confusion_matrix, print_classification_report
from classification.svm import load_or_train
from image.cell_extraction import extract_cells, plot_extracted_cells
from PIL import Image
import numpy as np
from image.feature_extraction import extract_features, load_data
from sklearn import cross_validation, datasets, svm
from sklearn.metrics import confusion_matrix
import pylab as pl

#from sklearn import datasets

# Choose a sudoku grid number, and prepare paths (image and verification grid)
sudoku_nb = 18
im_path = './data/ocr_data/{}.JPG'.format(sudoku_nb)
ver_path = './data/sudokus/sudoku{}.sud'.format(sudoku_nb)

im_npArray, label_npArray = load_data('./data/ocr_data/')
#for element in im_npArray:
#    print(element)

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(im_npArray, label_npArray, test_size=0.4, random_state=0)


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
# TODO: load the sudoku image as a gray level image
# file_path_string = tkFileDialog.askopenfilename()
# image = Image.open(file_path_string)

# Extract cells
# TODO

# Add data for each cell
# TODO: iterate over cells and append features to a list

# Classification
# TODO: use the classifier to predict on the list of features

# Load solution to compare with, print metrics, and print confusion matrix
y_sudoku = np.loadtxt(ver_path).reshape(81)
# TODO: print classification report
# TODO: show confusion matrix

# Print resulting sudoku
# TODO: print the resulting sudoku grid (use reshape() function to get a 9x9 grid print!
