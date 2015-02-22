import numpy as np
import matplotlib.pyplot as plt
from predict import *
import sys


plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
# set parameter of model in  caffe.Classifier


def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    
class Vizualization(object):
  """docstring for Vizualization"""
  def __init__(self, proto_path,bin_path):
    # Set the right path to your model definition file, pretrained model weights,
    # and the image you would like to classify.
    MODEL_FILE = proto_path 
    PRETRAINED = bin_path 

    self.net = caffe.Classifier (MODEL_FILE,PRETRAINED,
              mean=np.load('mean_file.npy'),
              channel_swap=(2,1,0),
              raw_scale=255,
              image_dims=(96, 96))
    self.net.set_phase_test()
    self.net.set_mode_gpu()
    
  def viz_image(self,path):
    scores = self.net.predict([caffe.io.load_image(path)])
    data = [(k, v.data.shape) for k, v in self.net.blobs.items()]
    data_2 = [(k, v[0].data.shape) for k, v in self.net.params.items()]
    plt.imshow(self.net.deprocess('data', self.net.blobs['data'].data[4]))
    # the parameters are a list of [weights, biases]
    filters = self.net.params['conv1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
    plt.show()
    
    feat = self.net.blobs['conv1'].data[4, :36]
    vis_square(feat, padval=1)
    plt.show()
    
    filters = self.net.params['conv2'][0].data
    vis_square(filters[:48].reshape(48**2, 5, 5))
    plt.show()