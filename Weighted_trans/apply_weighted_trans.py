from utils import *
import numpy as np
from scipy.optimize import minimize
from copy import deepcopy
from main_augmented import multiclass_log_loss
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import preprocessing

NUM_TRANS = 115
input_val = None

def extract_mean(data):
    mean = np.mean(data,axis = 0)
    return mean
  
  
def extract_weighted(data, weight):
     val = np.zeros((121,))
     sum_we = 0.0
     for i in range(NUM_TRANS):
       if weight[i] >  0.0:
        val += data[i,:] * weight[i]
        sum_we += float(weight[i])
     val  /= sum_we 
     return val

prediction = load_cPickle('amnie_val_all_list.npy')
weights_hand = np.load('weights_hand.npy')
labels     = np.load('labels.npy')
min_max_scaler = preprocessing.MinMaxScaler()
print weights_hand.shape
weights_scaled = min_max_scaler.fit_transform(weights_hand.reshape((115,1)))
#take vector from each predictions
list_pred = list()
for idx,val in enumerate(prediction):
  list_pred.append(extract_mean(val))
  
  
v = np.asarray(list_pred)
y_pred_label = np.argmax(v, axis=1)
print  "Normal ",multiclass_log_loss(labels, v), accuracy_score(labels, y_pred_label)

#apply weight
list_pred = list()
for idx,val in enumerate(prediction):
  list_pred.append(extract_weighted(val,weights_scaled))
  
  
v2 = np.asarray(list_pred)
y_pred_label = np.argmax(v, axis=1)
print  "Weighted ",multiclass_log_loss(labels, v2), accuracy_score(labels, y_pred_label)

weights_trans = np.load('weights_trans.npy')
weights_scaled = min_max_scaler.fit_transform(weights_trans.reshape((115,1)))
list_pred = list()
for idx,val in enumerate(prediction):
  list_pred.append(extract_weighted(val,weights_scaled))

v3 = np.asarray(list_pred)
y_pred_label = np.argmax(v, axis=1)
print  "Weighted ",multiclass_log_loss(labels, v3), accuracy_score(labels, y_pred_label)


model8 = np.load('probabilities_kaggle_test_8model.npy')
print  "Merge ",multiclass_log_loss(labels, (v+v2)/2)
