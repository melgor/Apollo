import numpy as np
import cv2
import os
# Make sure that caffe is on the python path:
caffe_root = '/home/blcv/LIB/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


class Prediction(object):
  """docstring for Prediction_Normalme"""
  def __init__(self, proto_path,bin_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path 
    PRETRAINED = bin_path 

    self.net = caffe.Classifier (MODEL_FILE,PRETRAINED,
              mean=np.load('mean_file.npy'),
              raw_scale= 255,
              image_dims=(96, 96),
              gpu=True)
              
      
  def predict(self, image):
    """Predict using Caffe normal model"""
    input_image = caffe.io.load_image(image,color=False)
    prediction = self.net.predict([input_image], oversample=False)
    return prediction
      
  def predict_multi(self, images):
    """Predict using Caffe normal model"""
    list_input = list()
    for image in images: 
      list_input.append(caffe.io.load_image(image,color=False))
    prediction = self.net.predict(list_input, oversample=False)
    return prediction
      

    
    
