# -*- coding: utf-8 -*-
# @Author: melgor
# @Date:   2014-11-27 11:54:32
# @Last Modified 2015-02-27
# @Last Modified time: 2015-02-27 22:43:31
import numpy as np
import json
import cPickle
import os
import gzip


#TODO add compressing using: Return 2x times smaller
#import bz2
        #import cPickle as pickle
        #with bz2.BZ2File('test.pbz2', 'w') as f:
            #pickle.dump(l, f)
def load_cPickle(name_file):
  '''Load file in cPickle format'''
  f = gzip.open(name_file,'rb')
  tmp = cPickle.load(f)
  f.close()
  return tmp

def save_cPickle(data, name_file):
  '''Save file in cPickle format, delete if exist'''
  if os.path.isfile(name_file):
      os.remove(name_file)
  f = gzip.open(name_file,'wb')
  cPickle.dump(data,f,protocol=2)
  f.close()

def create_dir(path):
  print path
  if not os.path.isdir(path):
    os.makedirs(path)