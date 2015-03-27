from tools import list_images
import numpy as np
from PIL import Image
from skimage.transform import resize


def extract_features(im):
    """ Returns a feature vector for an image patch. """

    # TODO: find other features to use
    return im.flatten()


def process_image(im, border_size=5, im_size=50):
    """ Remove borders and resize """

    im = im[border_size:-border_size, border_size:-border_size]
    im = resize(im, (im_size, im_size))

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




