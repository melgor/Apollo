The config_files folder has the following files and folders


1.) prototxt_template - this folder is for sample prototxt files, such as "solver", "model", and "deploy" files.  These are used by Caffe to train and predict.  Please add more samples, especially ones that have scored well on the LB.

2.) img_test.lst - this file is needed for creating a Kaggle submission.  You will need to replace the path with you own path to the TEST images for now.  A simple find and replace in Sublime worked for me.  Soon we should have a global fig file to take care of all of this hopefully.

