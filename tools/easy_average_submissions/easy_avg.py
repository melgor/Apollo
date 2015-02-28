import pandas as pd
import glob
import os


'''
This tool works as such. This file needs to be placed in a directory of Kaggle csv files you want to model average.  Example.  You have alexnet_submission.csv and googlenet_submission.csv and you want to average them.  You would place these 2 files into the same directory as this script and then run the script.  The script will save the averaged files to the same directory it is in.


    Note: Need to sort by column image before dropping it....this is because bartosz's files come in with jpgs in different orders as well as the columns can be out of order since the script got rid of the mapper file.

    To Do: you should add the ability to test this on your training/val data so don't over fit to test set.
'''

def average_submissions(submissions_list, out_file):
    '''
    #pre: takes in a list of Kaggle Submission files and name for an out_file, which is where the average of all the submissions will be saved.
    #post saves file to the provided file, out_file. nothing is returned.
    '''
    counter = 0
    df_list = []
    #loop through and get list of dataframes without image col.
    for sub_file in submissions_list:
        df1 = pd.read_csv(sub_file, header=0)
        #sort by images
        df1 = df1.sort('image', axis=0)
        df1 = df1.reset_index(drop=True)
        #save images col if first run.  Only need 1.
        if counter == 0:
            df_images1 = df1['image']
        #drop the images col
        df1 = df1.drop('image', axis=1)
        df_list.append(df1)
    #number of dfs to sum
    count = len(df_list)
    #print count
    
    #sum dfs and average them
    combined_dfs_avg = (sum(df_list)/count)
    #add back that image column that is strings
    df_merged = pd.concat([df_images1, combined_dfs_avg], axis=1)
    #print out_file
    df_merged.to_csv(out_file, index=False)

def get_csv_inDir():
    '''
    pre: place in directory with csv files you want to merge then run.
    post: returns list of all csv files in current directory.
    '''
    csv_list = []
    for files in glob.glob("*.csv"):
        #print files
        
        csv_list.append(files)

    #print csv_list
    
    return csv_list

#IF you want to manually provide the list of CSV files, you can do this here by uncommenting this.  Make sure you then comment out the line submissions_list = get_csv_inDir()
#submissions_list = ['sub1.csv', 'sub2.csv', 'sub3.csv']

#get current working directory
current_directory = os.path.dirname(os.path.abspath(__file__))
submissions_list = get_csv_inDir()
print 'Found these CSV files, preparing to merge them...', submissions_list



out_file = '%s/merged_submissions.csv' % current_directory

#print out_file

average_submissions(submissions_list, out_file)
print 'Saved merged CSV file to ',out_file