import glob
import os
import sys
import numpy as np
import datetime
import cPickle as pickle
from sklearn.utils import shuffle

###############################################################################

# Training set directory where 96x96 images are present (using gen_train.py)
FTRAIN = 'train_96x96_aug'
# File list directory
FLIST = '../Apollo/Data_Set'
# File prefix
FPREFIX = 'train_96x96_aug_'
# Sampling method
USE_CONFUSION_MATRIX = False
USE_RANDOM_SAMPLING = True

###############################################################################

directory_names = list(set(glob.glob(os.path.join(FTRAIN, "*"))).difference(set(glob.glob(os.path.join(FTRAIN,"*.*")))))

classMemberCount = dict()
desiredDistribution = dict()
desiredSampleCount = dict()
classes = list()
totalCount = 0
X_sample = list()

for folder in directory_names:
    className = folder.split('/')[-1]
    classes.append(className)
    classCount = 0
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            if fileName[-4:] != ".jpg":
              continue
            classCount += 1
            totalCount += 1
            
    classMemberCount[className] = classCount
    desiredDistribution[className] = 0.0 #init to zero

for currentClass in classes:
    desiredDistribution[currentClass] = classMemberCount[currentClass] / float(totalCount)


# Read class label mapping from saved pickled file
print "============================="
print "Reading class - label mapping"
print "============================="
with open(FLIST + '/' + FPREFIX + 'Labels.pickle', 'r') as f:
    savedLabels = pickle.load(f)
print ""

##############################################################################
## Example dataset for testing, keep these commented (START)
#desiredDistribution['acantharia_protist_halo'] = 0.98
#desiredDistribution['amphipods'] = 0.01
#desiredDistribution['copepod_other'] = 0.01
## Example confusion matrix
#confusionMatrix = [[10,  0,  0], 
#                   [0,  10,  0],
#                   [0,   0, 10]]
## Example dataset for testing, keep these commented (END)
##############################################################################


# Apply sampling method
print "========================"
print "Applying sampling method"
print "========================"
if (USE_CONFUSION_MATRIX == True) and (USE_RANDOM_SAMPLING == False):
    print "Sampling based on confusion matrix"
    print ""
    # Read confusion matrix
    #confusionMatrix = ... # (FILL THIS UP APPROPRIATELY)
    
    # Compute total number of samples
    sumRow = np.sum(confusionMatrix, axis = 1)
    
    # Compute number of misclassified samples
    sumMisclassification = sumRow - np.sum(np.eye(3) * confusionMatrix, axis = 1)
    
    # Adjust distribution
    for currentClass in classes:
        classLabel = savedLabels.index(currentClass)
        desiredDistribution[currentClass] += sumMisclassification[classLabel]/sumRow[classLabel]

elif (USE_RANDOM_SAMPLING == True) and (USE_CONFUSION_MATRIX == False):
    print "Sampling based on uniform random numbers"
    print ""
    for currentClass in classes:
        classLabel = savedLabels.index(currentClass)
        desiredDistribution[currentClass] = np.random.uniform(0.0, 1.0)
    
else:
    print "Error: One sampling method must be selected - USE_CONFUSION_MATRIX or USE_RANDOM_SAMPLING"
    print""
    sys.exit(0)
    
    
# Normalize desired distribution
print "================================"
print "Normalizing desired distribution"
print "================================"
sumProbability = 0.0
for currentClass in classes:
    sumProbability += desiredDistribution[currentClass]
for currentClass in classes:
    desiredDistribution[currentClass] /= sumProbability
print ""


# Determine the number of samples for each class based on the probability
print "==================================================="
print "Estimating sample count from specified distribution"
print "==================================================="
for currentClass in classes:
    desiredSampleCount[currentClass] = int(desiredDistribution[currentClass] * totalCount)
print ""


# Sample as per specified distibution
print "===================================="
print "Sampling from specified distribution"
print "===================================="
for currentClass in classes:
    folder = FTRAIN + '/' + currentClass
    classMember = list()
    for fileNameDir in os.walk(folder):
        for fileName in fileNameDir[2]:
            if fileName[-4:] != ".jpg":
                continue
            classMember.append(fileName)
                
        memberIdx = np.random.randint(0, classMemberCount[currentClass], size = desiredSampleCount[currentClass])
    
        for mIdx in memberIdx:
            X_sample.append(folder + '/' + currentClass + '/' + classMember[mIdx] + ' ' + str(savedLabels.index(currentClass)))
print ""


# Shuffle training and validation lists
print "==================="
print "Shuffling data sets"
print "==================="
X_sample = shuffle(X_sample, random_state = 0)
print ""


# Write train.txt
print "============================="
print "Writing sampled training list"
print "============================="
f = open(FLIST + '/' + FPREFIX + 'train_sample_' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')) + '.txt', 'wb')
f.writelines( "%s\n" % item for item in X_sample)
f.close()
print ""
