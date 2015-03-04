#!/usr/bin/env python
import numpy as np
import argparse,os
from collections import namedtuple
from utils import *
from sklearn.metrics import confusion_matrix, accuracy_score
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


def readfile(args):

  with open(args.feat_train, 'r') as f:
    file_train = [line for line in f]
  
  with open(args.feat_val, 'r') as f:
    file_val = [line for line in f]
    
  #read some train_data
  feat_train = list()
  shape = 0
  for i in range(0,3000):
    feat_train.append(load_cPickle(file_train[i].strip()))
    shape += feat_train[-1].data.shape[0]
  shape_x = feat_train[-1].data.shape[1]
  shape_y = feat_train[-1].data.shape[0]
  print shape_x, shape_y
  #extract feture vector and labels vector
  all_features_train = np.zeros((shape,shape_x))
  labels_train = list()
  for idx,elem in enumerate(feat_train):
    all_features_train[idx*shape_y : (idx+1)*shape_y,:] = elem.data
    labels_train.extend(elem.label)
  
  print    all_features_train.shape, len(labels_train)
  lab = np.asarray(labels_train)
  sgd = SGDClassifier(loss='log',n_jobs=6)
  sgd.fit(all_features_train, lab )
  save_cPickle(sgd,"sgd1.cPickle")
  rf =   RandomForestClassifier(n_jobs=6)
  rf.fit(all_features_train, lab )
  save_cPickle(rf,"rf1.cPickle")
  ada = AdaBoostClassifier(n_estimators=100)
  ada.fit(all_features_train, np.asarray(labels_train))
  save_cPickle(ada,"ada1.cPickle")
  #read some train_data
  feat_train = list()
  shape = 0
  for i in range(3000,len(file_train)):
    feat_train.append(load_cPickle(file_train[i].strip()))
    shape += feat_train[-1].data.shape[0]
  shape_x = feat_train[-1].data.shape[1]
  shape_y = feat_train[-1].data.shape[0]
  #extract feture vector and labels vector
  all_features_train = np.zeros((shape,shape_x))
  labels_train = list()
  for idx,elem in enumerate(feat_train):
    all_features_train[idx*shape_y : (idx+1)*shape_y,:] = elem.data
    labels_train.extend(elem.label)
  sgd2 =  SGDClassifier(loss='log',n_jobs=6)
  sgd2.fit(all_features_train, np.asarray(labels_train))
  save_cPickle(sgd2,"sgd2.cPickle")
  rf2 =   RandomForestClassifier(n_jobs=6)
  rf2.fit(all_features_train, np.asarray(labels_train))
  save_cPickle(rf2,"rf2.cPickle")
  ada2 = AdaBoostClassifier(n_estimators=100)
  ada2.fit(all_features_train, np.asarray(labels_train))
  save_cPickle(ada2,"ada2.cPickle")
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
    
    
  #merge result
  score = rf.score(all_features_val, labels_val)
  print score
  score = rf2.score(all_features_val, labels_val)
  print score
  
  score = sgd.score(all_features_val, labels_val)
  print score
  score = sgd2.score(all_features_val, labels_val)
  print score
  
  score = ada.score(all_features_val, labels_val)
  print score
  score = ada2.score(all_features_val, labels_val)
  print score
  
  predicted_rf1 = rf.predict_proba(all_features_val)
  predicted_rf2 = rf2.predict_proba(all_features_val)
  
  predicted_sgd1 = sgd.predict_proba(all_features_val)
  predicted_sgd2 = sgd2.predict_proba(all_features_val)
  
  predicted_agd1 = ada.predict_proba(all_features_val)
  predicted_agd2 = ada2.predict_proba(all_features_val)
  
  print "Rf:",multiclass_log_loss( np.asarray(labels_val), (predicted_rf1 + predicted_rf2)/2)
  print "SGD:",multiclass_log_loss( np.asarray(labels_val), (predicted_sgd1 + predicted_sgd2)/2)
  print "ADA:",multiclass_log_loss( np.asarray(labels_val), (predicted_agd1 + predicted_agd2)/2)
  
  print "all: ",multiclass_log_loss( np.asarray(labels_val), (predicted_agd1 + predicted_agd2 + predicted_sgd1 + predicted_sgd2 +predicted_rf1 + predicted_rf2 )/6)
  #score = rf.score(all_features_train, labels_train)
  #print score
  #p =  (predicted_1 + predicted_2)/2
  #y_pred_label = np.argmax(p, axis=1)
  #print accuracy_score(np.asarray(labels_val),y_pred_label)
    
  

  
if __name__ == '__main__':
  args        = parser.parse_args()
  readfile(args)
  
    