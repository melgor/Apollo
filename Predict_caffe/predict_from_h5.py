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

    self.net = caffe.Net (MODEL_FILE,PRETRAINED, caffe.TEST)


  def predict(self, h5file):
    """Predict using Caffe normal model"""
    print "File used ",h5file
    f = h5py.File(h5file.strip(), "r")
    input_features = f["data"].value
    data4D = np.zeros([128,1,1,5120])
    data4DL = np.zeros([128,1,1,1])
    data4D[:,0,0,:] = input_features
    self.net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
    pred = self.net.forward()
    pred['fc2'].shape = (128,121)
    predictions = pred['fc2']
    return predictions




