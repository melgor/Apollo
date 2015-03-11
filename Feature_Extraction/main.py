#!/usr/bin/env python
import numpy as np
import argparse,os
from extractor import Extractor
from collections import namedtuple
from utils import *

Feature = namedtuple('Feature', 'data label')

parser = argparse.ArgumentParser(description='Run Caffe model from dir and given label')
parser.add_argument('--images', required=True,help='path to file with paths to images')
parser.add_argument('--proto_path', required=True,help='path to proto file')
parser.add_argument('--bin_path', required=True,help='path to binary Net')
parser.add_argument('--folder', required=True,help='folder where save featues')


def extract_multi(args):
  # Extract  Featues from Net using prot_file from images from input file
  # it will save patch of max_value features at one .cPickle. 
  # Not as one file, because 2.4M images produce 10 GB of data
  pred = Extractor(args.proto_path,args.bin_path)
  max_value = 512 
  curr_value = 0
  list_all_result = list()
  list_good_class_all = list()
  list_name_file = list()
  create_dir(args.folder)
  with open(args.images,'r') as file_image:
    list_images = list()
    list_good_class = list()
    for idx,line in enumerate(file_image):
      splitted = line.split(' ')
      list_good_class.append(int(splitted[1]))
      list_images.append(splitted[0].strip())
      curr_value = curr_value + 1
      if curr_value < max_value:
        continue
      else:
        #predict using value
        predictions = pred.predict_multi(list_images)
        f = Feature(predictions,list_good_class)
        name = '/'.join((args.folder,str(idx)+"_file.cPickle"))
        list_name_file.append(os.path.abspath(name))
        save_cPickle(f,name)
        list_good_class = list()
        list_images = list()
        curr_value = 0
        print "Predicted 512"
        
    #predict last package of data, which is smaller than max_value
    if len(list_images) > 0:
      predictions = pred.predict_multi(list_images)
      list_all_result.append(predictions)
      f = Feature(predictions,list_good_class)
      name = '/'.join((args.folder,str(idx)+"_file.cPickle"))
      save_cPickle(f,name)
      list_name_file.append(os.path.abspath(name))
      
  f = open(args.folder+ '/' + 'files.txt', 'wb')
  f.writelines( "%s\n" % item for item in list_name_file)
  f.close()

  
if __name__ == '__main__':
  args        = parser.parse_args()
  extract_multi(args)
  
    