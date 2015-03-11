import numpy as np 
import sys 

all_data = None

def merge(x):
   '''Minimize loss merging probabilities'''
   sum_elem = np.sum(x)
   m = (x[0] * all_data[0] + x[1] * all_data[1] + x[2] * all_data[2] +x[3] * all_data[3] + x[4] * all_data[4] + x[5] * all_data[5] + x[6] * all_data[6] + x[7] * all_data[7] + x[8] * all_data[8] )/sum_elem
   return m
 
casia = np.load('merge_test/casia.npy')
casia_aug = np.load('merge_test/casia_aug.npy')
casia_cnn = np.load('merge_test/casia_cnn.npy')
casia_deep = np.load('merge_test/casia_deep.npy')
casia_drop = np.load('merge_test/casia_drop.npy')
casia_fl2 = np.load('merge_test/casia_fl2.npy')
casia_small = np.load('merge_test/casia_small.npy')
sand = np.load('merge_test/sand.npy') 
#using Amine script
casia_aug_a = np.load('merge_test_aug/casia_cnn_aug.npy')



weights = np.load(sys.argv[1])
all_data = [casia, casia_aug, casia_cnn, casia_deep, casia_drop, casia_fl2, casia_small, sand,casia_aug_a]

merged =  merge(weights)
np.savetxt('probabilities_kaggle_test_weighted.txt', merged, fmt='%1.9e')