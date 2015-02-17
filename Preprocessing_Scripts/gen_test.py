"""
Originally from https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl
"""

import os
import sys

if len(sys.argv) < 3:
    print "Usage: python gen_test.py input_folder output_folder size"
    exit(1)

fi = sys.argv[1]
fo = sys.argv[2]
sz = sys.argv[3]

cmd = "convert "
imgs = os.listdir(fi)


for img in imgs:
    md = ""
    md += cmd
    md += fi + img
    md += " -resize " + sz + "x" + sz + "\!"
    md += " " + fo + img
    os.system(md)



