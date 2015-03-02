import pandas as pd

def cap_min_prediction(prediction):
    if prediction < (1*10**-6):
        prediction = (1*10**-6)
    return prediction

def cap_max_prediction(prediction):
    if prediction > (.99):
        prediction = (.99)
    return prediction

df = pd.read_csv('/Users/me/Desktop/merge_4_model.csv',header=0)
#df2 = pd.read_csv('/Users/me/Desktop/out_pool_alex_f.csv',header=0)

#df_combined = (df1+df2)/2


#cap = (1*10**-10)
df = df.applymap(cap_min_prediction)
df.to_csv('/Users/me/Desktop/capTest_new.csv', index=False)