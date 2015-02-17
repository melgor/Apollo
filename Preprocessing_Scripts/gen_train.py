"""
Originally from https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl
"""

import os
import sys

if len(sys.argv) < 3:
    print "Usage: python gen_train.py input_folder output_folder size"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]
sz = sys.argv[3]

cmd = "convert "
classes = os.listdir(fi)

os.chdir(fo)
for cls in classes:
    try:
        os.mkdir(cls)
    except:
        pass
    print fi + cls
    imgs = os.listdir(fi + cls)
    for img in imgs:
        md = ""
        md += cmd
        md += fi + cls + "/" + img
        md += " -resize " + sz + "x" + sz + "\!"
        md += " " + fo + cls + "/" + img
        os.system(md)



