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
import pandas as pd
###############################################################################

# this is the location of the csv with the kaggle to blue label mapper
mapper_csv = '/Users/me/Desktop/kagle_info/ndsb/Apollo/config_files/classes_to_blue_labels.csv'
# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = '/Users/me/Desktop/kagle_info/ndsb/images/train_96x96/'
# File list directory
FLIST = '/Users/me/Desktop/kagle_info/ndsb/images/test_96x96'
# File prefix
FPREFIX = 'train_96x96_aug_blue_label'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL

###############################################################################

def get_blue_label(df_mapper, class_label):
    '''
    pre: Takes in a class label like amphipods and the mapper_df(which is the csv file imported as a Dataframe) and returns its blue label crustaceans.
    post:
    '''
    return df_mapper.loc[class_label].values[0]

def transform(mapper_df):
    """
    Tranform images in training set
    """

    X_train = list()
    X_val = list()
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
        # now, need to swap for blue label here...
        current_class = get_blue_label(df_mapper, current_class)
        
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
                    X_val.append(folder + '/' + fileName + ' ' + str(current_class_id))
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
        
    return X_train, X_val, X_train_files, X_val_files, files, classes

###############################################################################
#get the mapper file into a dataframe
df_mapper = pd.read_csv(mapper_csv, header=0, index_col=0)
# Apply transforms
print "========================"
print "Applying transformations"
print "========================"
X_train, X_val,X_train_files, X_val_files, files, classes = transform(df_mapper)
print ""

# Shuffle training and validation lists
print "==================="
print "Shuffling data sets"
print "==================="
X_train = shuffle(X_train, random_state = 0)
X_val = shuffle(X_val, random_state = 0)
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
