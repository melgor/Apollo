#!/usr/bin/env python
import numpy as np
import argparse,os
from predict import Prediction
from viz import Vizualization
parser = argparse.ArgumentParser(description='Run Caffe model from dir and given label')
parser.add_argument('--images', required=True,help='path to file with paths to images')
parser.add_argument('--proto_path', required=True,help='path to proto file')
parser.add_argument('--bin_path', required=True,help='path to binary Net')
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
  
# test multiple images in one batch. Batch size set by max_value. Same value should be at deploy file
def test_accuracy_multi(args):
  pred = Prediction(args.proto_path,args.bin_path)
  num_images = 0
  good_images = 0
  max_value = 512 
  curr_value = 0
  list_all_result = list()
  list_good_class_all = list()
  
  with open(args.images,'r') as file_image:
    list_images = list()
    list_good_class = list()
    for line in file_image:
      splitted = line.split('\t')

      list_good_class.append(int(splitted[1]))
      list_images.append(splitted[0].strip())
      curr_value = curr_value + 1
      if curr_value < max_value:
        continue
      else:
        #predict using value
        predictions = pred.predict_multi(list_images)
        list_all_result.append(predictions)
        pred_value  = np.argmax(predictions, axis=1)
        print pred_value.shape
        for i in range(0,max_value):
          if pred_value[i] == list_good_class[i]:
            good_images = good_images + 1 
          num_images = num_images + 1
        list_images = list()
        list_good_class_all.extend(list_good_class)
        list_good_class = list()
        curr_value = 0
        
    #predict using value
    if len(list_images) > 0:
      predictions = pred.predict_multi(list_images)
      list_all_result.append(predictions)
      pred_value  = np.argmax(predictions, axis=1)
      print pred_value.shape
      for i in range(0,len(list_images)):
        if pred_value[i] == list_good_class[i]:
          good_images = good_images + 1
        num_images = num_images + 1
          
  print "Print:", float(good_images)/num_images
  #test loss
  list_good_class_all.extend(list_good_class)
  y_truth = np.asarray(list_good_class_all)
  y_pred = np.vstack(list_all_result)
  print y_truth.shape, y_pred.shape
  print multiclass_log_loss(y_truth,y_pred)


#predict each image separately
def predict_image(args):
  pred = Prediction(args.proto_path,args.bin_path)
  all_img = 0
  good_img = 0
  with open(args.images,'r') as file_image:
    for line in file_image:
      line_v = line.split(' ')
      print line_v[0]
      predictions = pred.predict(line_v[0])
      pred_value  = np.argmax(predictions, axis=1)
      if pred_value[0] == int(line_v[1].strip()):
        good_img +=1 
      all_img += 1
  
  print 'Accuracy: ', good_img/float(all_img)

#extract probabilities of each class. Need for submission file. 
#it use multi-prediction code
def prob_image(args):
  list_predictions = list()
  pred = Prediction(args.proto_path,args.bin_path)
  max_value = 512 
  curr_value = 0

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
        list_images = list()
        list_good_class = list()
        curr_value = 0
        list_predictions.append(predictions)
        print predictions.shape
        
    if len(list_images) > 0:
      #predict using value
      predictions = pred.predict_multi(list_images)
      list_predictions.append(predictions)
  
  all_pred = np.vstack(list_predictions)
  np.savetxt('out_pred.txt', all_pred, fmt='%1.9e')
  
if __name__ == '__main__':
  args        = parser.parse_args()
  if args.extract_prob:
    prob_image(args)
  elif args.viz != None:
    viz = Vizualization(args.proto_path,args.bin_path)
    viz.viz_image(args.viz)
  else:
    test_accuracy_multi(args)
  
    