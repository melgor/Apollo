#!/usr/bin/env python
import numpy as np
import argparse,os
from predict import Prediction
from viz import Vizualization
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.pylab import cm

parser = argparse.ArgumentParser(description='Run Caffe model from dir and given label')
parser.add_argument('--images', required=True,help='path to file with paths to images')
parser.add_argument('--proto_path', required=True,help='path to proto file')
parser.add_argument('--bin_path', required=True,help='path to binary Net')
parser.add_argument('--mapper', required=False,help='mapper between labels, use when create submission')
parser.add_argument('--extract_prob', required=False,action="store_true",help='extract prob and save to file')
parser.add_argument('--viz', required=False,help='vizualize featues, give path to image')

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
  
def plot_confusion_matrix(true_labels,pred_labels):
    # Compute confusion matrix and save to drive
    conf_mat = confusion_matrix(np.array(true_labels), np.array(pred_labels))
    print(conf_mat)
    # Show confusion matrix in a separate window
    plt.matshow(conf_mat, cmap=cm.jet)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('Conf_Matrix.png', bbox_inches='tight')
  
def test_accuracy_multi(args):
  # Compute Accuracy of images from input file
  # Code predict many images at one time. The value is set by max_value
  # Same value should be set in proto file
  # It is done because of speed reason
  
  pred = Prediction(args.proto_path,args.bin_path)
  max_value = 512 
  curr_value = 0
  list_all_result = list()
  list_good_class_all = list()
  if args.mapper != None:
    #create mapping between our labels and Kaggle labels
    with open(args.mapper) as data_file:    
      maper = json.load(data_file)
  
    good_list = list()
    for i in range(121):
      key_value  = 0
      for key,elem in maper.iteritems():
        if i == elem:
          key_value =  key
          break
      good_list.append(int(key))
    
  with open(args.images,'r') as file_image:
    list_images = list()
    list_good_class = list()
    for line in file_image:
      splitted = line.split('\t')
      #if not os.path.isfile(splitted[0].strip()):
        #continue
      list_good_class.append(int(splitted[1]))
      list_images.append(splitted[0].strip())
      curr_value = curr_value + 1
      if curr_value < max_value:
        continue
      else:
        #predict using value
        predictions = pred.predict_multi(list_images)
        if args.mapper != None:
          pred_good = predictions[:,good_list]
        else:
          pred_good = predictions
        list_all_result.append(pred_good)
        list_good_class_all.extend(list_good_class)
        list_good_class = list()
        list_images = list()
        curr_value = 0
        
    #predict last package of data, which is smaller than max_value
    if len(list_images) > 0:
      predictions = pred.predict_multi(list_images)
      if args.mapper != None:
          pred_good = predictions[:,good_list]
      else:
          pred_good = predictions
      list_all_result.append(pred_good)
          
  #test loss
  list_good_class_all.extend(list_good_class)
  y_truth = np.asarray(list_good_class_all)
  y_pred = np.vstack(list_all_result)
  y_pred_label = np.argmax(y_pred, axis=1)
  print y_truth.shape, y_pred.shape
  print "Accuracy: ", accuracy_score(y_truth,y_pred_label)
  print "Loss: ",multiclass_log_loss(y_truth,y_pred)
  plot_confusion_matrix(y_truth,y_pred_label)
 
  
  

def prob_image(args):
  list_predictions = list()
  pred = Prediction(args.proto_path,args.bin_path)
  max_value = 512 
  curr_value = 0
  with open(args.mapper) as data_file:    
    maper = json.load(data_file)
    
  good_list = list()
  for i in range(121):
    key_value  = 0
    for key,elem in maper.iteritems():
      if i == elem:
        key_value =  key
        break
    good_list.append(int(key))


  with open(args.images,'r') as file_image:
    list_images = list()
    list_good_class = list()
    for line in file_image:
      splitted = line.split('\t')
      list_images.append(splitted[2].strip())
      curr_value = curr_value + 1
      if curr_value < max_value:
        continue
      else:
        #predict using value
        predictions = pred.predict_multi(list_images)
        pred_good = predictions[:,good_list]
        list_images = list()
        list_good_class = list()
        curr_value = 0
        list_predictions.append(pred_good)
        print predictions.shape
        
    if len(list_images) > 0:
      #predict using value
      predictions = pred.predict_multi(list_images)
      pred_good = predictions[:,good_list]
      list_predictions.append(pred_good)
  
  all_pred = np.vstack(list_predictions)
  #np.save('pred.npy',all_pred)
  np.savetxt('probabilities_kaggle.txt', all_pred, fmt='%1.9e')
  
if __name__ == '__main__':
  args        = parser.parse_args()
  if args.extract_prob:
    prob_image(args)
  elif args.viz != None:
    print "Vizualization"
    viz = Vizualization(args.proto_path,args.bin_path)
    viz.viz_image(args.viz)
  else:
    test_accuracy_multi(args)
  
    