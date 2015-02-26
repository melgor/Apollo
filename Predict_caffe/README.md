# Prediction using Caffe

This scripts allows to use Caffe model to test any file with paths to images or to extract probalibities of each class (used for submission).

ATTENTION: Most of script point to Caffe Python directory, you need to change it firstly

To run it, firstly you need:
 1. Binary model (endswitch= .caffemodel)
 
 2. Prototxt file: proper prototxt file, which should be created in deploy manner (see example in caffe/models/bvlc_reference_caffenet/deploy.prototxt)
 
 3. Create mean_file.npy, should be place in same folder like main. Do it by:
  python bin_to_np.py dim_image_mean path_to_mean_caffe name_output

The main.py is used for prediction and it has following arguments:
  - images       : path to file with pathes to images ex. produced by augmented script
  - proto_path   : path to deploy file for Caffe model
  - bin_path     : path to caffe model
  - extract_prob : use model for extracting probabilities from images for each sample. Save it as csv file
  
  
For running caffe model for prediction or creating submission:
 1. Run prediction using main. There are two ways of use it: test accuracy/loss or extract    probalibities for Kaggle (Filter Vizualization need more changes). For testing file you need ti run:
  ./main.py --images path_to_file  --proto_path path_to_deploy --bin_path path_to_binary_model
  That script produce accuracy value, loss value and confusion matrix.
 
Create file for submission:
  1. Run prediction script. Mapper is needed to transform our labels to Kaggle labels
  ./main.py --images path_to_file  --proto_path path_to_deploy --bin_path path_to_binary_model --extract_prob
  This script will produce "probabilities_kaggle.txt" file, which can be used for creating submission
 
 2. After createing "probabilities_kaggle.txt" file, to create .csv file use "make_submission.py"
  python make_submission.py mapper.pickle img_test.lst probabilities_kaggle.txt out.csv
  
  "mapper.pickle" have to be taken from Sandipto script, which was used for data augmentation

 