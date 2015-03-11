import glob
import os
import sys
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
FTRAIN = '/home/ubuntu/ndsb/images/train_96x96/'
# File list directory
FLIST = '/home/ubuntu/ndsb/images/train_96x96'
# File prefix
FPREFIX = 'train_96x96_aug_'
# Pixel size for final image
MAX_IMAGE_PIXEL = 96
IMAGE_SIZE = MAX_IMAGE_PIXEL * MAX_IMAGE_PIXEL

###############################################################################

def transform(image_file):
   
    X_train = list()
    with open(image_file,'r') as f:
      for line in f:
        splitted = line.strip().split(' ' )
        current_class_id = splitted[1]
        path = splitted[0]
        print "Reading: " + path + " (" + str(current_class_id) + ")"
        
        if path[-4:] != ".jpg":
            continue
        #print fileName
        
        
        # ===================================
        # The original file, no modifications
        # ===================================
        image = sc.misc.imread(path)
        X_train.append(line.strip())
     
        
        # ======================
        # Scale 1 original image
        # ======================
        similarity_transform = SimilarityTransform(scale = 0.75)
        image_scaled = warp(image, similarity_transform, mode = 'wrap')
        new_path  = path.split('.jpg')[0]  + '_scale1.jpg'
        sc.misc.imsave(new_path , image_scaled)
        X_train.append(new_path +' ' + str(current_class_id))
      

        # ======================
        # Scale 2 original image
        # ======================
        similarity_transform = SimilarityTransform(scale = 1.25)
        image_scaled = warp(image, similarity_transform, mode = 'wrap')
        new_path  = path.split('.jpg')[0]  + '_scale2.jpg'
        sc.misc.imsave(new_path, image_scaled)
        X_train.append(new_path + ' ' + str(current_class_id))
     

        # =======================================
        # Rotate image by intervals of 45 degrees
        # =======================================
        for degrees in [45, 90, 135, 180, 225, 270, 315]:

            # =========================
            # Rotate image by 'degrees'
            # =========================
            image_rotated = rotate(image, degrees, mode = 'wrap')
            image_resized = np.resize(image_rotated, (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL))
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '.jpg'
            sc.misc.imsave(new_path, image_resized)
            X_train.append(new_path + ' ' + str(current_class_id))
          

            # =========================================
            # On the current rotated image, perform ...
            # =========================================

            # ==========================
            # Scale 1 (on rotated image)
            # ==========================
            similarity_transform = SimilarityTransform(scale = 0.75)
            image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_scale1.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
      

            # ==========================
            # Scale 2 (on rotated image)
            # ==========================
            similarity_transform = SimilarityTransform(scale = 1.25)
            image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_scale2.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
         

            # ==========================
            # Shear 1 (on rotated image)
            # ==========================
            affine_transform = AffineTransform(shear = 0.2)
            image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_sheared1.jpg'
            sc.misc.imsave(new_path, image_sheared)
            X_train.append(new_path + ' ' + str(current_class_id))
          

            # ============================================
            # Shear 1 Scale 1 (on rotated - sheared image)
            # ============================================
            similarity_transform = SimilarityTransform(scale = 0.75)
            image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_shear1_scale1.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
        

            # ============================================
            # Shear 1 Scale 2 (on rotated - sheared image)
            # ============================================
            similarity_transform = SimilarityTransform(scale = 1.25)
            image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_shear1_scale2.jpg'
            sc.misc.imsave(new_path, image_scaled)          
            X_train.append(new_path + ' ' + str(current_class_id))
        
            # ==========================
            # Shear 2 (on rotated image)
            # ==========================
            affine_transform = AffineTransform(shear = -0.2)
            image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_sheared2.jpg'
            sc.misc.imsave(new_path, image_sheared)
            X_train.append(new_path + ' ' + str(current_class_id))
            
            
            # ============================================
            # Shear 2 Scale 1 (on rotated - sheared image)
            # ============================================
            similarity_transform = SimilarityTransform(scale = 0.75)
            image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_shear2_scale1.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
            

            # ============================================
            # Shear 2 Scale 2 (on rotated - sheared image)
            # ============================================
            similarity_transform = SimilarityTransform(scale = 1.25)
            image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_shear2_scale2.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
            

            # ==============================
            # Translate 1 (on rotated image)
            # ==============================
            similarity_transform = SimilarityTransform(translation = (0, 20))
            image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_translate1.jpg'
            sc.misc.imsave(new_path, image_translated)
            X_train.append(new_path + ' ' + str(current_class_id))
          
            
            # ==============================
            # Translate 2 (on rotated image)
            # ==============================
            similarity_transform = SimilarityTransform(translation = (20, 0))
            image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_translate2.jpg'
            sc.misc.imsave(new_path, image_translated)
            X_train.append(new_path + ' ' + str(current_class_id))
            
            
            # ==============================
            # Translate 3 (on rotated image)
            # ==============================
            similarity_transform = SimilarityTransform(translation = (-20, 0))
            image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_translate3.jpg'
            sc.misc.imsave(new_path, image_translated)
            X_train.append(new_path + ' ' + str(current_class_id))
            
            
            # ==============================
            # Translate 4 (on rotated image)
            # ==============================
            similarity_transform = SimilarityTransform(translation = (0, -20))
            image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_translate4.jpg'
            sc.misc.imsave(new_path, image_translated)
            X_train.append(new_path + ' ' + str(current_class_id))
            
            
            # Scale 1
            similarity_transform = SimilarityTransform(scale = 0.75)
            image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_scale1.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
      

            # Scale 2
            similarity_transform = SimilarityTransform(scale = 1.25)
            image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
            new_path  = path.split('.jpg')[0] + '_' + str(degrees) + '_scale2.jpg'
            sc.misc.imsave(new_path, image_scaled)
            X_train.append(new_path + ' ' + str(current_class_id))
          

        
        
    return X_train

###############################################################################

# Apply transforms
print "========================"
print "Applying transformations"
print "========================"
X_train = transform(sys.argv[1])
print ""


# Write train.txt
print "====================="
print "Writing training list"
print "====================="
f = open(FPREFIX + 'train.txt', 'wb')
f.writelines( "%s\n" % item for item in X_train )
f.close()
print ""

