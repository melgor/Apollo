import numpy as np
import cv2
import os
# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/repositories/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import h5py


class PredictionFromH5(object):
  """docstring for Prediction_Normalme"""
  def __init__(self, proto_path,bin_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path
    PRETRAINED = bin_path

    self.net = caffe.Classifier (MODEL_FILE,PRETRAINED,
              image_dims=(2000,),
              gpu=True)


  def predict(self, h5file):
    """Predict using Caffe normal model"""
    f = h5py.File(h5file, "r")
    input_features = f["data"].values()
    prediction = self.net.predict([input_features], oversample=False)
    return prediction




