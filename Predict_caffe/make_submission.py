import csv
import sys
import cPickle as pickle

#make submission file. For it you need sample submission, path to test images, result.
if len(sys.argv) < 4:
    print "Usage: python make_submission.py labels.pickle test.lst text.txt out.csv"
    exit(1)

with open(sys.argv[1],'r') as f:
    lab_our = pickle.load(f)


#this is the test.lst file.
fl = csv.reader(file(sys.argv[2]), delimiter='\t', lineterminator='\n')

#this is the kaggle probabilities file.
fi = csv.reader(file(sys.argv[3]), delimiter=' ', lineterminator='\n')
fo = csv.writer(open(sys.argv[4], "w"), lineterminator='\n')

#first column is the image file name
head = ['image']
#add in the classes in the order we created the test data.
head.extend(lab_our)
#write the column headers to the submission file.
fo.writerow(head)


img_lst = []
for line in fl:
    path = line[-1]
    path = path.split('/')
    path = path[-1]
    img_lst.append(path)

idx = 0
for line in fi:
    row = [img_lst[idx]]
    idx += 1
    line = line#[:-1]
    row.extend(line)
    fo.writerow(row)

