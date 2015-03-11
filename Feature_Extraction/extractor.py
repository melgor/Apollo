import numpy as np
import cv2
import os
# Make sure that caffe is on the python path:
caffe_root = '/home/blcv/LIB/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe


class Extractor(caffe.Net):
  """docstring for Prediction_Normalme"""
  def __init__(self, proto_path,bin_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path 
    PRETRAINED = bin_path 
    
    caffe.set_phase_test()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    caffe.set_device(0)
    caffe.Net.__init__(self, MODEL_FILE, PRETRAINED)
    self.set_raw_scale(self.inputs[0], 255.0)
    self.set_mean(self.inputs[0], np.load('mean_file.npy'))
    self.image_dims  = (96, 96)
    self.crop_dims = np.array(self.blobs[self.inputs[0]].data.shape[2:])
    
 
  def predict_multi(self, images):
    """Predict using Caffe normal model"""
    list_input = list()
    for image in images: 
      list_input.append(caffe.io.load_image(image,color=False))
    
    input_ = np.zeros((len(list_input),
            self.image_dims[0], self.image_dims[1], list_input[0].shape[2]),
            dtype=np.float32)
    for ix, in_ in enumerate(list_input):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)
    
    # Take center crop.
    center = np.array(self.image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -self.crop_dims / 2.0,
        self.crop_dims / 2.0
    ])
    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

    # run network
    caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = self.preprocess(self.inputs[0], in_)
    out = self.forward_all(**{self.inputs[0]: caffe_in})
    predictions = out[self.outputs[0]].squeeze(axis=(2,3))
    
    return predictions.astype(np.float16)
      

    
    
