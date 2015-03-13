#!/usr/bin/env python
import numpy as np
import argparse,os
from predict_from_h5 import PredictionFromH5

parser = argparse.ArgumentParser(description='Run Caffe model from dir and given label')
parser.add_argument('--features', required=True,help='path to file with paths to hdf files')
parser.add_argument('--proto_path', required=True,help='path to proto file')
parser.add_argument('--bin_path', required=True,help='path to binary Net')


def prob_image(args):
  list_predictions = list()
  pred = PredictionFromH5(args.proto_path,args.bin_path)

  with open(args.features,'r') as h5_list:
    for line in h5_list:
      print "Predicting from file ",line
      current = pred.predict(line)
      list_predictions.append(current)
  all_pred = np.vstack(list_predictions)
  np.save('fine_sand_h5.npy',all_pred)
  np.savetxt('probabilities_kaggle_aug_h5.txt', all_pred, fmt='%1.9e')


if __name__ == '__main__':
  args        = parser.parse_args()
  prob_image(args)
