import numpy as np
import pandas as pd
import scipy
import sklearn

print(np.__version__)
print(pd.__version__)
print(scipy.__version__)
print(sklearn.__version__)

crop_recommend_df = pd.read_csv(filepath_or_buffer="datasets/Crop_recommendation.csv")
#print(df.head())

test_df = crop_recommend_df.drop(["label"], axis = 1)

test_df_1_col = test_df["rainfall"]
#print(test_df.head())

df_shape = test_df.shape
#print(df_shape)
#print(type(df_shape))
#print(df_shape[0])
#print(df_shape[1])

count = 0
sum = 0
for row in range(df_shape[0]):
    count += 1
    sum += test_df_1_col.loc[row]

test_df_mean = sum / count

