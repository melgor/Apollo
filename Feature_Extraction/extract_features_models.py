from extractor import *
import numpy as np
from collections import namedtuple
import h5py

Feature = namedtuple('Feature', 'data label')
LIST_MODELS = [('amine__iter_20000.caffemodel', 'amine.prototxt')]
OUTPUT_FOLDER = "./combined_features"
images = "train_96x96_aug_val.txt"


def predict_and_combine(extractors, list_images):
    features = []
    for ext in extractors.values():
        features.append(ext.predict_multi(list_images))
    features = np.concatenate(features, axis=1)
    return features


def write_features_h5file(features, labels, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=features, **comp_kwargs)
        f.create_dataset('label', data=labels.astype(np.float32), **comp_kwargs)
    with open(OUTPUT_FOLDER + "/features_files_list.txt", 'a') as f:
        f.write(filename + "\n")

def extract_from_models():
    extractors = {}
    for bin_path, proto_path in LIST_MODELS:
        ext = Extractor(proto_path, bin_path)
        extractors[bin_path.split('.')[0]] = ext
    max_value = 512
    curr_value = 0
    list_all_result = list()
    list_good_class_all = list()
    list_name_file = list()
    with open(images, 'r') as file_image:
        list_images = list()
        list_good_class = list()
        for idx, line in enumerate(file_image):
            splitted = line.split(' ')
            list_good_class.append(int(splitted[1]))
            list_images.append(splitted[0].strip())
            curr_value = curr_value + 1
            if curr_value < max_value:
                continue
            else:
		print "Predicting for batch at index ",idx
                # predict using value
                features = predict_and_combine(extractors, list_images)
                name = '/'.join((OUTPUT_FOLDER, str(idx) + "_file.h5py"))
                list_name_file.append(os.path.abspath(name))
                write_features_h5file(
                    features, np.array(list_good_class), name)
                list_good_class = list()
                list_images = list()
                curr_value = 0

        # predict last package of data, which is smaller than max_value
        if len(list_images) > 0:
            features = predict_and_combine(list_images)
            name = '/'.join((OUTPUT_FOLDER, str(idx) + "_file.cPickle"))
            write_features_h5file(features, np.array(list_good_class), name)
            list_name_file.append(os.path.abspath(name))

if __name__ == "__main__":
	extract_from_models()
