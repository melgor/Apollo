import pandas as pd

'''
You need to change the paths to your own directoris.

Edit the cap variables to play around with them.
'''

path_in_csv = '/Users/me/Desktop/merge_4_model.csv'
path_out_csv = '/Users/me/Desktop/capTest_new.csv'
min_cap = .005
max_cap = .99


def cap_min_prediction(prediction):
    if prediction < min_cap:
        prediction = min_cap
    return prediction


def cap_max_prediction(prediction):
    if isinstance(prediction, str):
        pass
    else:
        if prediction > max_cap:
            prediction = max_cap
    return prediction

df = pd.read_csv(path_in_csv, header=0)

# print df.describe()

# comment out if you only want a max or min cap.
df = df.applymap(cap_min_prediction)
df = df.applymap(cap_max_prediction)

df.to_csv(path_out_csv, index=False)
