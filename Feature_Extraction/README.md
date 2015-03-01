# Extract Featues using CAFFE

This scripts allows to use Caffe model for features extraction

ATTENTION: Most of script point to Caffe Python directory, you need to change it firstly

Features are saved in pack of 512. Mainly because 2.4M images take ~10 GB. 
Features are saved using float16 to reduce space too. 
Each package is the tuple with data(features) and label(corresponding labels)

To run it, firstly you need:
 1. Binary model (endswitch= .caffemodel)
 
 2. Prototxt file: proper prototxt file, which should be created in deploy manner (see example in caffe/models/bvlc_reference_caffenet/deploy.prototxt). It should point the layer which you would like to use for extracting. Ex. if you would like to extract features from "ip2", you should remove all layer after it (beyond ReLu layer, which should stay)
 
 3. Create mean_file.npy, should be place in same folder like main. Do it by:
  python bin_to_np.py dim_image_mean path_to_mean_caffe name_output

The main.py is used for prediction and it has following arguments:
  - images       : path to file with pathes to images ex. produced by augmented script
  - proto_path   : path to deploy file for Caffe model
  - bin_path     : path to caffe model
  - folder       : where save features
  

  
 ./main.py --images path_to_file  --proto_path path_to_deploy --bin_path path_to_binary_model --folder path_to_folder

 