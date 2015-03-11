from utils import *
import numpy as np
from scipy.optimize import minimize,fminbound
from copy import deepcopy

NUM_TRANS = 115
input_val = None

def min_loss(x):
     sum_val = 0
     new_mat = np.zeros(input_val.shape)
     for i in range(NUM_TRANS):
       new_mat[:,i] = input_val[:,i] *  x[i]
       
     sum_rows = np.sum(new_mat,axis = 1)  
     sum_cols = np.sum(sum_rows,axis = 0)
     #print sum_cols/np.sum(x)
     return -sum_cols/(115*121*3034)



def extract_vec(data,label):
  good_class_pred = data[:,label]
  #print good_class_pred.shape, good_class_pred.T.shape
  #print sum(good_class_pred.T), sum((1-good_class_pred.T))
  return (good_class_pred.T)
  

prediction = load_cPickle('amnie_val_all_list.npy')
labels     = np.load('labels.npy')
print len(prediction)

#take vector from each predictions
list_pred = list()
for idx,val in enumerate(prediction):
  list_pred.append(extract_vec(val,labels[idx]))
  
  


input_val = np.asarray(list_pred)

np.savetxt('proba.txt', input_val, fmt='%1.9e')

print input_val.shape
x0 = np.ones((1,115))
#print min_loss(x0)
res = minimize(min_loss, x0, method='nelder-mead',
                options={'xtol': 1e-8, 'disp': True, 'maxiter' : 1000000000, 'maxfev': 1000000000})


print min_loss(res.x)
w = res.x
np.save("weights_trans.npy", w)

