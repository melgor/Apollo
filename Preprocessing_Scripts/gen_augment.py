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

###############################################################################

# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = '/home/blcv/CODE/Plancton/Test_96/train_aug_all/'
# File list directory
FLIST = '/home/blcv/CODE/Plancton/Test_96/train_aug_all/'
# File prefix
FPREFIX = 'train_96x96_aug_'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL

###############################################################################

def transform():
    """
    Tranform images in training set
    """
    X_train = list()
    directory_names = list(set(glob.glob(os.path.join(FTRAIN, "*"))).difference(set(glob.glob(os.path.join(FTRAIN,"*.*")))))

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
    X_train_files, X_val_files = train_test_split(files, test_size = 0.1, random_state = 0)
    
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
                
                image_file = "{0}{1}{2}".format(fileNameDir[0], os.sep, fileName)
                
                # ===================================
                # The original file, no modifications
                # ===================================
                image = sc.misc.imread(image_file)
                
                if (folder + '/' + fileName) in X_train_files:
                    X_train.append(folder + '/' + fileName + ' ' + str(current_class_id))
                else:
                    continue
                
                # ======================
                # Scale 1 original image
                # ======================
                similarity_transform = SimilarityTransform(scale = 0.75)
                image_scaled = warp(image, similarity_transform, mode = 'wrap')
                sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_scale1.jpg', image_scaled)
                if (folder + '/' + fileName) in X_train_files:
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_scale1.jpg' + ' ' + str(current_class_id))
                else:
                    X_val.append(folder + '/' + fileName.split('.')[0] + '_scale1.jpg' + ' ' + str(current_class_id))

                # ======================
                # Scale 2 original image
                # ======================
                similarity_transform = SimilarityTransform(scale = 1.25)
                image_scaled = warp(image, similarity_transform, mode = 'wrap')
                sc.misc.imsave(folder + '/' + fileName.split('.')[0]  + '_scale2.jpg', image_scaled)
                if (folder + '/' + fileName) in X_train_files:
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_scale2.jpg' + ' ' + str(current_class_id))
                else:
                    X_val.append(folder + '/' + fileName.split('.')[0] + '_scale2.jpg' + ' ' + str(current_class_id))

                # =======================================
                # Rotate image by intervals of 45 degrees
                # =======================================
                for degrees in [45, 90, 135, 180, 225, 270, 315]:

                    # =========================
                    # Rotate image by 'degrees'
                    # =========================
                    image_rotated = rotate(image, degrees, mode = 'wrap')
                    image_resized = np.resize(image_rotated, (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL))
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '.jpg', image_resized)
                   
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '.jpg' + ' ' + str(current_class_id))
                 

                    # =========================================
                    # On the current rotated image, perform ...
                    # =========================================

                    # ==========================
                    # Scale 1 (on rotated image)
                    # ==========================
                    similarity_transform = SimilarityTransform(scale = 0.75)
                    image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg', image_scaled)
                    if (folder + '/' + fileName) in X_train_files:
                        X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg' + ' ' + str(current_class_id))
                    else:
                        X_val.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg' + ' ' + str(current_class_id))

                    # ==========================
                    # Scale 2 (on rotated image)
                    # ==========================
                    similarity_transform = SimilarityTransform(scale = 1.25)
                    image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg', image_scaled)
                    if (folder + '/' + fileName) in X_train_files:
                        X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg' + ' ' + str(current_class_id))
                    else:
                        X_val.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg' + ' ' + str(current_class_id))

                    # ==========================
                    # Shear 1 (on rotated image)
                    # ==========================
                    affine_transform = AffineTransform(shear = 0.2)
                    image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared1.jpg', image_sheared)
                    
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared1.jpg' + ' ' + str(current_class_id))
                  

                    # ============================================
                    # Shear 1 Scale 1 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale = 0.75)
                    image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale1.jpg', image_scaled)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale1.jpg' + ' ' + str(current_class_id))
                

                    # ============================================
                    # Shear 1 Scale 2 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale = 1.25)
                    image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale2.jpg', image_scaled)
                  
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale2.jpg' + ' ' + str(current_class_id))
                
                    # ==========================
                    # Shear 2 (on rotated image)
                    # ==========================
                    affine_transform = AffineTransform(shear = -0.2)
                    image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared2.jpg', image_sheared)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared2.jpg' + ' ' + str(current_class_id))
                   
                    
                    # ============================================
                    # Shear 2 Scale 1 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale = 0.75)
                    image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale1.jpg', image_scaled)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale1.jpg' + ' ' + str(current_class_id))
                   

                    # ============================================
                    # Shear 2 Scale 2 (on rotated - sheared image)
                    # ============================================
                    similarity_transform = SimilarityTransform(scale = 1.25)
                    image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale2.jpg', image_scaled)
                    
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale2.jpg' + ' ' + str(current_class_id))
                   

                    # ==============================
                    # Translate 1 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(translation = (0, 20))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate1.jpg', image_translated)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate1.jpg' + ' ' + str(current_class_id))
                 
                    
                    # ==============================
                    # Translate 2 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(translation = (20, 0))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate2.jpg', image_translated)

                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate2.jpg' + ' ' + str(current_class_id))
                   
                    
                    # ==============================
                    # Translate 3 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(translation = (-20, 0))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate3.jpg', image_translated)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate3.jpg' + ' ' + str(current_class_id))
                   
                    
                    # ==============================
                    # Translate 4 (on rotated image)
                    # ==============================
                    similarity_transform = SimilarityTransform(translation = (0, -20))
                    image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate4.jpg', image_translated)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate4.jpg' + ' ' + str(current_class_id))
                    
                    
                    # Scale 1
                    similarity_transform = SimilarityTransform(scale = 0.75)
                    image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg', image_scaled)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg' + ' ' + str(current_class_id))
             

                    # Scale 2
                    similarity_transform = SimilarityTransform(scale = 1.25)
                    image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
                    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg', image_scaled)
                    X_train.append(folder + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg' + ' ' + str(current_class_id))
          

        current_class_id += 1
        
    return X_train, X_train_files, X_val_files, files, classes

###############################################################################

# Apply transforms
print "========================"
print "Applying transformations"
print "========================"
X_train, X_train_files, X_val_files, files, classes = transform()
print ""

# Shuffle training and validation lists
print "==================="
print "Shuffling data sets"
print "==================="
X_train = shuffle(X_train, random_state = 0)
X_val = shuffle(X_val_files, random_state = 0)
print ""

# Write train.txt
print "====================="
print "Writing training list"
print "====================="
f = open(FLIST + '/' + FPREFIX + 'train.txt', 'wb')
f.writelines( "%s\n" % item for item in X_train )
f.close()
print ""

# Write val.txt
print "======================="
print "Writing validation list"
print "======================="
f = open(FLIST + '/' + FPREFIX + 'val.txt', 'wb')
f.writelines( "%s\n" % item for item in X_val)
f.close()
print ""

# Save class labels
print "===================="
print "Storing class labels"
print "===================="
with open(FLIST + '/' + FPREFIX + 'Labels.pickle', 'wb') as f:
    pickle.dump(classes, f)
print ""
