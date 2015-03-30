from tools import list_images
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import rank
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, skeletonize
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure


def extract_features(im):
    """ Returns a feature vector for an image patch. """

    # TODO: find other features to use
    img = img_as_float(im)
    hist  = img.ravel();
    return hist.flatten()


def process_image(im, border_size=8, im_size=50):
    """ Remove borders and resize """

    sx, sy = im.shape;

    # noise removal
    im = median(im, disk(2))

    # binarization
    thresh = threshold_otsu(im)
    im = im > thresh
	  
    # erosion
    im = erosion(im, disk(1))

    im = im[ sx/8:-sx/8, sy/8:-sy/8]

    im = resize(im , (im_size, im_size))

    # Si la case est presque blanche
    if (np.mean(im) >= 0.95) :
	im[:,:] = 1.0
    elif (np.mean(im) <= 0.1) :
	im[:,:] = 1.0
    return im


def load_data(path):
    """ Return labels and features for all jpg images in path. """
    
    # Create a list of all files ending in .jpg
    im_list = list_images(path, '.jpg')

    # Create labels
    labels = [int(im_name.split('/')[-1][0]) for im_name in im_list]
    
    # for x in range(0,200):
    #     print(im_list[x])
    #     print(labels[x])
    # Create feature list
    feature_list = list() 

    # Create features from the images
    # TODO: iterate over images paths
    for element in im_list:
        # TODO: load image as a gray level image
        img = np.array(Image.open(element).convert('L'))
        # TODO: process the image to remove borders and resize
        img = process_image(img)
        # TODO: append extracted features to the a list
        feature_list.append(extract_features(img))
	

    # TODO: return features, and labels
    return np.array(feature_list), np.array(labels)




