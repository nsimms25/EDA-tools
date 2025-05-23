import numpy as np
import pandas as pd
import scipy
import sklearn
from math import sqrt

crop_recommend_df = pd.read_csv(filepath_or_buffer="datasets/Crop_recommendation.csv")
#print(df.head())

test_df = crop_recommend_df.drop(["label"], axis = 1)

test_df_1_col = test_df[["rainfall"]]
#print(test_df.head())

df_shape = test_df.shape
#print(df_shape)
#print(type(df_shape))
#print(df_shape[0])
#print(df_shape[1])

def my_count(dataframe):
    count = len(dataframe)
    return count

def my_sum(dataframe):
    row_count = dataframe.shape[0]
    func_count = my_count(dataframe)

    sum = 0
    for row in range(row_count):
        sum += dataframe.loc[row]
    
    return pd.to_numeric(sum)

def my_mean(dataframe):
    row_count = dataframe.shape[0]
    func_count = my_count(dataframe)
    func_sum = my_sum(dataframe)

    mean = func_sum / func_count

    return mean

def my_stdev(dataframe):
    row_count = dataframe.shape[0]
    func_count = my_count(dataframe)
    func_mean = my_mean(dataframe)

    df = dataframe.iloc[:, 0]
    values = pd.to_numeric(df, errors='coerce').dropna()

    sq_diffs = sum((x - func_mean) ** 2 for x in values)
    variance = sq_diffs / (func_count - 1)

    return variance ** 0.5

print("=COUNT=")
print(my_count(test_df_1_col))
print(len(test_df_1_col))
print("=SUM=")
print(my_sum(test_df_1_col))
print(np.sum(test_df_1_col, axis=0))
print("=MEAN=")
print(my_mean(test_df_1_col))
print(np.mean(test_df_1_col, axis=0))
print("=STD=")
print(my_stdev(test_df_1_col))
print(np.std(test_df_1_col, axis=0))

