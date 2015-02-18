import glob
import os
import numpy as np
import scipy as sc
from skimage.transform import resize
from skimage import morphology
from skimage import measure
from sklearn import preprocessing
import cPickle as pickle

###############################################################################

# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = 'train_96x96'
# Test set directory where 96x96 images are present (using gen_test.py)
FTEST = 'test_96x96'
# Output directory for preprocessed training images
FOUT_TRAIN = 'train_96x96'
# Output directory for preprocessed test images
FOUT_TEST = 'test_96x96'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL
# Ignore the following
#MAX_PATCH_PIXEL = 8
#PATCH_SIZE = MAX_PATCH_PIXEL * MAX_PATCH_PIXEL
#NUM_PATCH = (MAX_IMAGE_PIXEL/MAX_PATCH_PIXEL) ** 2
#DEBUG = False

###############################################################################

def getLargestRegion(regions, imlabeled, imthr):
    """
    From Kaggle tutorial for NDSB
    https://www.kaggle.com/c/datasciencebowl/details/tutorial
    """
    regionmaxprop = None
    for regionprop in regions:
        # check to see if the region is at least 10% nonzero
        if sum(imthr[imlabeled == regionprop.label])*1.0/regionprop.area < 0.1:
            continue
        if regionmaxprop is None:
            regionmaxprop = regionprop
        if regionmaxprop.filled_area < regionprop.filled_area:
            regionmaxprop = regionprop
    
    return regionmaxprop

###############################################################################

def load(test = False):
    """
    Load training or test sets, preprocess and create arrays
    """
    if test == False:    
        directory_names = list(set(glob.glob(os.path.join(FTRAIN, "*"))).difference(set(glob.glob(os.path.join(FTRAIN,"*.*")))))
    else:
        directory_names = list(set(glob.glob(os.path.join(FTEST))))

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
    
    #X = np.zeros((n_images * NUM_PATCH, PATCH_SIZE))
    #X = np.zeros((n_images, IMAGE_SIZE))
    #y = np.zeros((n_images * NUM_PATCH))
    #Y = np.zeros(n_images)
    image_idx = 0
    current_class_id = 0
    
    for folder in directory_names:
        current_class = folder.split(os.sep)[-1]
        print "Reading: " + current_class + " (" + str(current_class_id) + ")"
        classes.append(current_class)
        for fileNameDir in os.walk(folder):
            for fileName in fileNameDir[2]:
                if fileName[-4:] != ".jpg":
                    continue
                #print fileName
                #f = plt.figure(figsize=(8,2))
                image_file = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                image = sc.misc.imread(image_file)
                #sub1 = plt.subplot(1, 4, 1)
                #sub1.set_title('Original')                
                #plt.imshow(image_copy, cmap=cm.gray)

                image_thr = np.where(image > np.mean(image), 0, 1)
                #sub2 = plt.subplot(1, 4, 2)
                #sub2.set_title('Threshold image')
                #plt.imshow(image_thr, cmap=cm.gray)

                image_dilated = morphology.dilation(image_thr.astype(np.uint8), np.ones((2, 2)))
                image_labeled = measure.label(image_dilated)
                image_regions = measure.regionprops(image_labeled)
                maxregion = getLargestRegion(image_regions, image_labeled, image_thr)

                if maxregion == None:
                    #print "Segmentation failed: " + fileName
                    mask = np.where(image_thr == 0.0)
                else:
                    image_maxregion = np.where(image_labeled == maxregion.label, 1.0, 0.0)
                    image_maxregion = morphology.erosion(image_maxregion.astype(np.uint8), np.ones((2, 2)))
                    mask = np.where(image_maxregion == 0.0)

                image[mask] = 0.0
                image = resize(image, (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL))
                #sub3 = plt.subplot(1, 4, 3)
                #sub3.set_title('Extracted region')
                #plt.imshow(image, cmap=cm.gray)

                min_max_scaler = preprocessing.MinMaxScaler()
                image = min_max_scaler.fit_transform(image)  
                #sub4 = plt.subplot(1, 4, 4)
                #sub4.set_title('After min-max scale')
                #plt.imshow(image, cmap=cm.gray)


                #X[image_idx] = np.reshape(image, (1, IMAGE_SIZE))
                #patch = view_as_blocks(image, block_shape = (MAX_PATCH_PIXEL, MAX_PATCH_PIXEL))
                #
                #patch_idx = 0
                #for ii in range(MAX_IMAGE_PIXEL/MAX_PATCH_PIXEL):
                #    for jj in range(MAX_IMAGE_PIXEL/MAX_PATCH_PIXEL):
                #        temp = patch[ii, jj, :, :].reshape(1, PATCH_SIZE)
                #        #print temp
                #        X[image_idx + patch_idx, :] = temp #np.reshape(image, (1, IMAGE_SIZE))
                #        patch_idx += 1

                #y[image_idx:image_idx + NUM_PATCH] = current_class_id
                #Y[image_idx] = current_class_id
                #image_idx += NUM_PATCH
                image_idx += 1
                
                if test == False:                
                    if not os.path.exists(FOUT_TRAIN + '/' + current_class):
                        os.makedirs(FOUT_TRAIN + '/' + current_class)
                    sc.misc.imsave(FOUT_TRAIN + '/' + current_class + '/' + fileName, image)
                else:
                    if not os.path.exists(FOUT_TEST):
                        os.makedirs(FOUT_TEST)
                    sc.misc.imsave(FOUT_TEST + '/' + fileName, image)
                    
        current_class_id += 1
                
    return n_images, n_classes, directory_names, classes#, X, Y


###############################################################################

# Load training set
print "======================="
print "Processing training set"
print "======================="
_, _, _, _, _, _ = load(test = False)
print ""

## Save training set
#with open('X_train.pickle', 'wb') as f:
#    pickle.dump(X_train, f)

## Save training set labels
#with open('Y_train.pickle', 'wb') as f:
#    pickle.dump(Y_train, f)

## Save class names
#with open('class_train.pickle', 'wb') as f:
#    pickle.dump(classes_train, f)

# Load test set
print "==================="
print "Processing test set"
print "==================="
_, _, _, _, _, _ = load(test = True)
print ""

## Save testset
#with open('X_test.pickle', 'wb') as f:
#    pickle.dump(X_test, f)

# To display images, keep disabled
#for idx in range(200):
#    plt.imshow(np.reshape(X_train[idx, :], (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL)), cmap=cm.gray)
#    plt.show()
