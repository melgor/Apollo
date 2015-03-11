#!/usr/bin/env python
import numpy as np
import argparse,os
from collections import namedtuple
from utils import *
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier 
Feature = namedtuple('Feature', 'data label')

parser = argparse.ArgumentParser(description='Create classifier using features from Net')
parser.add_argument('--feat_train', required=True,help='path to file with paths to train fetures')
parser.add_argument('--feat_val', required=True,help='path to file with paths to validation features')



def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
  
  
def learn_classifiers(data,labels,idx, list_classifier):
  sgd = SGDClassifier(loss='log',n_jobs=6)
  sgd.fit(data, labels )
  save_cPickle(sgd,"sgd" + str(idx) +".cPickle")
  list_classifier.append([sgd,"SGD"])
  rf =   RandomForestClassifier(n_jobs=6)
  rf.fit(data, labels )
  save_cPickle(rf,"rf" + str(idx) +".cPickle")
  list_classifier.append([rf,"RF"])

def readfile(args):

  with open(args.feat_train, 'r') as f:
    file_train = [line for line in f]
  
  with open(args.feat_val, 'r') as f:
    file_val = [line for line in f]
    
  #read some train_data
  change_ratio = 1000
  idx_seg = 0
  list_classifier = list()
  for v in range(0,6000,change_ratio):
    feat_train = list()
    shape = 0
    for i in range(v,v+change_ratio):
      feat_train.append(load_cPickle(file_train[i].strip()))
      shape += feat_train[-1].data.shape[0]
    shape_x = feat_train[-2].data.shape[1]
    shape_y = feat_train[-2].data.shape[0]
    print shape_x, shape_y
    #extract feture vector and labels vector
    all_features_train = np.zeros((shape,shape_x))
    labels_train = list()
    for idx,elem in enumerate(feat_train):
      all_features_train[idx*shape_y : (idx+1)*shape_y,:] = elem.data
      labels_train.extend(elem.label)
    
    print   all_features_train.shape, len(labels_train)
    lab = np.asarray(labels_train)
    learn_classifiers(all_features_train,labels_train,idx_seg,list_classifier)
    idx_seg += 1
  

  #read valdata
  feat_val = list()
  shape = 0
  for elem in file_val:
    feat_val.append(load_cPickle(elem.strip()))
    shape += feat_val[-1].data.shape[0]

  #extract feture vector and labels vector
  all_features_val = np.zeros((shape,shape_x))
  labels_val = list()
  for idx,elem in enumerate(feat_val):
    all_features_val[idx*shape_y : (idx+1)*shape_y,:] = elem.data
    labels_val.extend(elem.label)
    
  
  for cl in list_classifier:
    #merge result
    score = cl[1].score(all_features_val, labels_val)
    print "Accuracy: ", cl[0], score
    print "Loss:",cl[0],multiclass_log_loss( np.asarray(labels_val))
    
    
  

  
if __name__ == '__main__':
  args        = parser.parse_args()
  readfile(args)
  
    