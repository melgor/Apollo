import glob
import os
import numpy as np
import scipy as sc
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import rotate
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
import cPickle as pickle

'''
This is a modification of Sandipto's script gen_augment.py.  Instead of saving each transformation, we are going to combine the images and transformed images together.

This is how he is saving them.  Instead we should try stacking the images. arr : ndarray, MxN ....the paramter is a numpy array.
http://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imsave.html

Will probably need to adjust this since they will be stacked.

# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL
A step that needs to be checked and could be thrown by aug this script is this...
# Save class labels
not sure if this will be correct if you only save 1 image at a time...consider just saving it back in place? so you don't need to keep track of it.
'''
###############################################################################

# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = '/home/blcv/CODE/Plancton/Test_96/train_aug_all/'
# File list directory
FLIST = '/home/blcv/CODE/Plancton/Test_96/train_aug_all/'
# File prefix
FPREFIX = 'train_stacked_aug_'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96

# this doesn't appear to be used again?
#IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL

###############################################################################


def transform():
    """
    Tranform images in training set
    """
    X_train = list()
    X_val = list()
    directory_names = list(set(glob.glob(os.path.join(
        FTRAIN, "*"))).difference(set(glob.glob(os.path.join(FTRAIN, "*.*")))))

    n_images = 0
    n_classes = 0
    y = list()
    classes = list()
    files = list()

    for folder in directory_names:
        y.append(n_classes)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                 # Only read in the image files
                if fileName[-4:] != ".jpg":
                    continue
                files.append(folder + '/' + fileName)
                n_images += 1
        n_classes += 1

    # Determine the training and validation split
    # Files or transformations of the files in the validation set must not be in the train set
    # Validation set must be completely independent of the training set
    X_train_files, X_val_files = train_test_split(
        files, test_size=0.1, random_state=0)

    current_class_id = 0

    for folder in directory_names:
        current_class = folder.split(os.sep)[-1]
        print "Reading: " + current_class + " (" + str(current_class_id) + ")"
        classes.append(current_class)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                if fileName[-4:] != ".jpg":
                    continue
                # print fileName

                image_file = "{0}{1}{2}".format(
                    fileNameDir[0], os.sep, fileName)

                # ===================================
                # The original file, no modifications
                # ===================================
                '''
                WILL PROBABLY STACK THEN SAVE IN THE SAME PLACE
                '''
                image = sc.misc.imread(image_file)
                # end original image

                # =======================================
                # Rotate image by intervals of 45 degrees
                # =======================================
                for degrees in [0, 45, 90, 135, 180, 225, 270, 315]:

                    # =========================
                    # Rotate image by 'degrees' but don't rotate on the first image.
                    # =========================
                    if rotate != 0:
                        image_rotated_1 = rotate(image, degrees, mode='wrap')
                    else:
                        image_rotated_1 = image
                    # ==========================
                    # Scale 1 (on rotated image)
                    # ==========================
                    similarity_transform = SimilarityTransform(scale=0.75)
                    image_scaled_2 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')
                    # ==========================
                    # Scale 2 (on rotated image)
                    # ==========================
                    similarity_transform = SimilarityTransform(scale=1.25)
                    image_scaled_3 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')
                    # ==========================
                    # Shear 1 (on rotated image)
                    # ==========================
                    affine_transform = AffineTransform(shear=0.2)
                    image_sheared_4 = warp(
                        image_rotated_1, affine_transform, mode='wrap')
                    # ============================================
                    # Shear 1 Scale 1 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale=0.75)
                    image_scaled_5 = warp(
                        image_sheared_4, similarity_transform, mode='wrap')
                    # ============================================
                    # Shear 1 Scale 2 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale=1.25)
                    image_scaled_6 = warp(
                        image_sheared_4, similarity_transform, mode='wrap')
                    # ==========================
                    # Shear 2 (on rotated image)
                    # ==========================
                    affine_transform = AffineTransform(shear=-0.2)
                    image_sheared_7 = warp(
                        image_rotated_1, affine_transform, mode='wrap')
                    # ============================================
                    # Shear 2 Scale 1 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale=0.75)
                    image_scaled_8 = warp(
                        image_sheared_7, similarity_transform, mode='wrap')
                    # ============================================
                    # Shear 2 Scale 2 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale=1.25)
                    image_scaled_9 = warp(
                        image_sheared_7, similarity_transform, mode='wrap')
                    # ==============================
                    # Translate 1 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(
                        translation=(0, 20))
                    image_translated_10 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')
                    # ==============================
                    # Translate 2 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(
                        translation=(20, 0))
                    image_translated_11 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')
                    # ==============================
                    # Translate 3 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(
                        translation=(-20, 0))
                    image_translated_12 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')
                    # ==============================
                    # Translate 4 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(
                        translation=(0, -20))
                    image_translated_13 = warp(
                        image_rotated_1, similarity_transform, mode='wrap')

                    # CHRIS ADDED THIS IN  now you stack and save here....

                    # YOU HAVE 14 images that need to be stacked....then saved.
                    # Also, you might need to put them in the train vs.
                    # validation folder.  See the notes on this.

                    sc.misc.imsave(
                        folder + '/' + fileName.split('.')[0], image_scaled)

                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(
                        degrees) + '_scale2.jpg' + ' ' + str(current_class_id))

        current_class_id += 1

    return X_train, X_val, X_train_files, X_val_files, files, classes

###############################################################################

# Apply transforms
print "========================"
print "Applying transformations"
print "========================"
X_train, X_val, X_train_files, X_val_files, files, classes = transform()
print ""

# Shuffle training and validation lists
print "==================="
print "Shuffling data sets"
print "==================="
X_train = shuffle(X_train, random_state=0)
X_val = shuffle(X_val, random_state=0)
print ""

# Write train.txt
print "====================="
print "Writing training list"
print "====================="
f = open(FLIST + '/' + FPREFIX + 'train.txt', 'wb')
f.writelines("%s\n" % item for item in X_train)
f.close()
print ""

# Write val.txt
print "======================="
print "Writing validation list"
print "======================="
f = open(FLIST + '/' + FPREFIX + 'val.txt', 'wb')
f.writelines("%s\n" % item for item in X_val)
f.close()
print ""

# Save class labels
print "===================="
print "Storing class labels"
print "===================="
with open(FLIST + '/' + FPREFIX + 'Labels.pickle', 'wb') as f:
    pickle.dump(classes, f)
print ""
