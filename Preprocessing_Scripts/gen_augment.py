import glob
import os
import numpy as np
import scipy as sc
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import rotate
from sklearn.cross_validation import train_test_split

###############################################################################

# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = 'train_96x96'
# File list directory
FLIST = '../Apollo/Data_Set'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL

###############################################################################

def transform():
    """
    Tranform images in training set
    """
    X_path_label = list()
    directory_names = list(set(glob.glob(os.path.join(FTRAIN, "*"))).difference(set(glob.glob(os.path.join(FTRAIN,"*.*")))))

    n_images = 0
    n_classes = 0
    y = list()
    classes = list()
    
    for folder in directory_names:
        y.append(n_classes)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                 # Only read in the image files
                if fileName[-4:] != ".jpg":
                  continue
                n_images += 1
        n_classes += 1
    
    current_class_id = 0
    
    for folder in directory_names:
        current_class = folder.split(os.sep)[-1]
        print "Reading: " + current_class + " (" + str(current_class_id) + ")"
        classes.append(current_class)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                if fileName[-4:] != ".jpg":
                    continue
                print fileName
                #f = plt.figure(figsize=(8,2))
                image_file = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                image = sc.misc.imread(image_file)

                X_path_label.append(folder + '/' + fileName + ' ' + str(current_class_id))                                

                # Rotate image by intervals of 45 degrees
                for degrees in [45, 90, 135, 180, 225, 270, 315]:
                    # total number of images = 1 (original) + 7 (rotation angle) * 7 (transformation) = 50
                    # Rotation                    
                    image_rotated = rotate(image, degrees, mode = 'wrap')
                    image_resized = np.resize(image_rotated, (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL))
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '.jpg', image_resized)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '.jpg' + ' ' + str(current_class_id))

                    # Shear 1
                    affine_transform = AffineTransform(shear = 0.2)
                    image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared1.jpg', image_sheared)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared1.jpg' + ' ' + str(current_class_id))
                    
                    # shear 2
                    affine_transform = AffineTransform(shear = -0.2)
                    image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared2.jpg', image_sheared)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared2.jpg' + ' ' + str(current_class_id))
                    
                    # Translate 1
                    similarity_transform = SimilarityTransform(translation = (0, 20))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate1.jpg', image_translated)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate1.jpg' + ' ' + str(current_class_id))
                    
                    # Translate 2
                    similarity_transform = SimilarityTransform(translation = (20, 0))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate2.jpg', image_translated)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate2.jpg' + ' ' + str(current_class_id))
                    
                    # Translate 3
                    similarity_transform = SimilarityTransform(translation = (-20, 0))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate3.jpg', image_translated)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate3.jpg' + ' ' + str(current_class_id))
                    
                    # Translate 4
                    similarity_transform = SimilarityTransform(translation = (0, -20))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate4.jpg', image_translated)
                    X_path_label.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate4.jpg' + ' ' + str(current_class_id))
                    
        current_class_id += 1
        
    return X_path_label

###############################################################################

# Apply transforms
X_path_label = transform()

# Split into test and validation sets
X_train, X_val = train_test_split(X_path_label, test_size = 0.1, random_state = 0)

# Write train.txt
f = open(FLIST + '/' + 'train.txt', 'wb')
f.writelines( "%s\n" % item for item in X_train )
f.close()

# Write val.txt
f = open(FLIST + '/' + 'val.txt', 'wb')
f.writelines( "%s\n" % item for item in X_val)
f.close()
