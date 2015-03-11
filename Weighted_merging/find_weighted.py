import numpy as np
from merge import *
from scipy.optimize import minimize

all_data = None
def rosen(x):
     """The Rosenbrock function"""
     return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
   
   
def min_loss(x):
   '''Minimize loss merging probabilities'''
   sum_elem = np.sum(x)
   m = (x[0] * all_data[0] + x[1] * all_data[1] + x[2] * all_data[2] +x[3] * all_data[3] + x[4] * all_data[4] + x[5] * all_data[5] + x[6] * all_data[6] + x[7] * all_data[7] + x[8] * all_data[8] )/sum_elem
   return multiclass_log_loss(all_data[9],m)
   

casia = np.load('merge/casia.npy')
casia_aug = np.load('merge/casia_aug.npy')
casia_cnn = np.load('merge/casia_cnn.npy')
casia_deep = np.load('merge/casia_deep.npy')
casia_drop = np.load('merge/casia_drop.npy')
casia_fl2 = np.load('merge/casia_fl2.npy')
casia_small = np.load('merge/casia_small.npy')
sand = np.load('merge/sand.npy') 

#casia_a = np.load('merge_aug/casia.npy')
casia_aug_a = np.load('merge_aug/casia_aug_aug.npy')
#casia_cnn_a = np.load('merge_aug/casia_cnn.npy')
#casia_drop_a = np.load('merge_aug/casia_drop_aug.npy')
#casia_fl2_a = np.load('merge_aug/casia_fl2.npy') #bad
##casia_small_a = np.load('merge_aug/casia_small_aug.npy')
#sand_a = np.load('merge_aug/sand.npy') 


labels = np.load('merge/labels.npy')

all_data = [casia, casia_aug, casia_cnn, casia_deep, casia_drop, casia_fl2, casia_small, sand,casia_aug_a, labels]
x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0])
res = minimize(min_loss, x0, method='nelder-mead',
                options={'xtol': 1e-8, 'disp': True, 'maxiter': 250})


print min_loss(res.x)
np.save("weights_2.npy",res.x)


  
#min_loss = 1.0
#best_weight = None
#for a in xrange(0, 100, 10):
  #for b in range(0, 100, 10):
    #for c in range(0, 100, 10):
      #for d in range(0, 100, 10):
        #for e in range(0, 100, 10):
          #for f in range(0, 100, 10):
            #for g in range(0, 100, 10):
              #for h in range(0, 100, 10):
                  #sum_elem = a + b + c + d + e + f + g + h
                  #m = (a * casia + b * casia_aug + c * casia_cnn + d * casia_deep + e * casia_drop + f * casia_fl2 + g * casia_small + h * sand)/sum_elem
                  #val_loss = multiclass_log_loss(labels,m)
                  #if val_loss < min_loss:
                    #min_loss = val_loss
                    #print min_loss
                    #best_weight = [a,b,c,d,e,f,g,h]
                    
#print "Min loss: ",    min_loss  , best_weight     //0.680201276938
           