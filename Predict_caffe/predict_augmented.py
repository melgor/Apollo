import numpy as np
import cv2
import os
# Make sure that caffe is on the python path:
caffe_root = '/home/blcv/LIB/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
from skimage.io import imread
from skimage.transform import AffineTransform
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.transform import rotate
import glob
print glob.glob("/home/adam/*.txt")

def generate_transformations(image, filename, folder):
  # if the transformations do not exist,
  # we create the folder that holds them,
  # and then create the transformations
  if not os.path.exists(folder + '/' + fileName.split('.')[0]):
    os.makedirs(folder + '/' + fileName.split('.')[0])

    transformed_images = [image]
    # ======================
    # Scale 1 original image
    # ======================
    similarity_transform = SimilarityTransform(scale = 0.75)
    image_scaled = warp(image, similarity_transform, mode = 'wrap')
    transformed_images.append(image_scaled)
    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_scale1.jpg', image_scaled)

    # ======================
    # Scale 2 original image
    # ======================
    similarity_transform = SimilarityTransform(scale = 1.25)
    image_scaled = warp(image, similarity_transform, mode = 'wrap')
    transformed_images.append(image_scaled)
    sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0]  + '_scale2.jpg', image_scaled)

    # =======================================
    # Rotate image by intervals of 45 degrees
    # =======================================
    for degrees in [45, 90, 135, 180, 225, 270, 315]:

      # =========================
      # Rotate image by 'degrees'
      # =========================
      image_rotated = rotate(image, degrees, mode = 'wrap')
      transformed_images.append(image_rotated)
      image_resized = np.resize(image_rotated, (MAX_IMAGE_PIXEL, MAX_IMAGE_PIXEL))
      transformed_images.append(image_resized)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '.jpg', image_resized)


      # =========================================
      # On the current rotated image, perform ...
      # =========================================

      # ==========================
      # Scale 1 (on rotated image)
      # ==========================
      similarity_transform = SimilarityTransform(scale = 0.75)
      image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg', image_scaled)

      # ==========================
      # Scale 2 (on rotated image)
      # ==========================
      similarity_transform = SimilarityTransform(scale = 1.25)
      image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg', image_scaled)

      # ==========================
      # Shear 1 (on rotated image)
      # ==========================
      affine_transform = AffineTransform(shear = 0.2)
      image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
      transformed_images.append(image_sheared)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared1.jpg', image_sheared)


      # ============================================
      # Shear 1 Scale 1 (on rotated - sheared image)
      # ============================================
      similarity_transform = SimilarityTransform(scale = 0.75)
      image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale1.jpg', image_scaled)


      # ============================================
      # Shear 1 Scale 2 (on rotated - sheared image)
      # ============================================
      similarity_transform = SimilarityTransform(scale = 1.25)
      image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear1_scale2.jpg', image_scaled)id))

      # ==========================
      # Shear 2 (on rotated image)
      # ==========================
      affine_transform = AffineTransform(shear = -0.2)
      image_sheared = warp(image_rotated, affine_transform, mode = 'wrap')
      transformed_images.append(image_sheared)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_sheared2.jpg', image_sheared)


      # ============================================
      # Shear 2 Scale 1 (on rotated - sheared image)
      # ============================================
      similarity_transform = SimilarityTransform(scale = 0.75)
      image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale1.jpg', image_scaled)


      # ============================================
      # Shear 2 Scale 2 (on rotated - sheared image)
      # ============================================
      similarity_transform = SimilarityTransform(scale = 1.25)
      image_scaled = warp(image_sheared, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_shear2_scale2.jpg', image_scaled)


      # ==============================
      # Translate 1 (on rotated image)
      # ==============================
      similarity_transform = SimilarityTransform(translation = (0, 20))
      image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_translated)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate1.jpg', image_translated)


      # ==============================
      # Translate 2 (on rotated image)
      # ==============================
      similarity_transform = SimilarityTransform(translation = (20, 0))
      image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_translated)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate2.jpg', image_translated)


      # ==============================
      # Translate 3 (on rotated image)
      # ==============================
      similarity_transform = SimilarityTransform(translation = (-20, 0))
      image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_translated)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate3.jpg', image_translated)


      # ==============================
      # Translate 4 (on rotated image)
      # ==============================
      similarity_transform = SimilarityTransform(translation = (0, -20))
      image_translated = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_translated)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_translate4.jpg', image_translated)


      # Scale 1
      similarity_transform = SimilarityTransform(scale = 0.75)
      image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale1.jpg', image_scaled))


      # Scale 2
      similarity_transform = SimilarityTransform(scale = 1.25)
      image_scaled = warp(image_rotated, similarity_transform, mode = 'wrap')
      transformed_images.append(image_scaled)
      sc.misc.imsave(folder + '/' + fileName.split('.')[0] + '/' + fileName.split('.')[0] + '_' + str(degrees) + '_scale2.jpg', image_scaled)

  else:
    transformed_images = []
    transformed_images_filenames = glob.glob(folder + '/' + fileName.split('.')[0])
    for image_file in transformed_images_filenames:
      transformed_images.append(imread(image_file))

  return transformed_images


class PredictionAugmented(object):
  """docstring for Prediction_Normalme"""
  def __init__(self, proto_path,bin_path,transformed_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path
    PRETRAINED = bin_path
    self.transformed_path = transformed_path

    self.net = caffe.Classifier (MODEL_FILE,PRETRAINED,
              mean=np.load('mean_file.npy'),
              raw_scale= 255,
              image_dims=(96, 96),
              gpu=True)


  def predict(self, image):
    """Predict using Caffe normal model"""
    input_image = caffe.io.load_image(image,color=False)
    transformed_images = generate_transformations(input_image, image, self.transformed_path)
    predictions = self.net.predict(transformed_images, oversample=False)
    prediction = np.mean(predictions, axis=0)
    return prediction

  def predict_multi(self, images):
    """Predict using Caffe normal model"""
    predictions = []
    for image in images:
      predictions.append(self.predict(image))
    return np.array(predictions)


