import pandas as pd
import numpy as np
import scipy.stats

def my_mean(dataframe):
    """
    This is my own creation for mean, this will not be useful over pandas or numpy.
    This is for statistics practice in python.

    Args:
        dataframe (DataFrame): Pandas DataFrame 
    
    Returns:
        Mean (float): mean statistic for the dataframe input.
    """
    df_shape = dataframe.shape
    row_count = df_shape[0]
    column_count = df_shape[1]
    count = 0 
    sum = 0
    for i in range(row_count):
        count += 1
        sum += dataframe.loc[i]
    
    return sum / count 

crop_recommend_df = pd.read_csv(filepath_or_buffer="datasets/Crop_recommendation.csv")
#print(df.head())

test_df = crop_recommend_df[["rainfall"]]
#print(test_df.head())

def get_summary_stats(dataframe):
    return dataframe.describe().T.assign(
        missing_pct=dataframe.isnull().mean() * 100,
        skewness=dataframe.skew(),
        kurtosis=dataframe.kurt()
    )

#print(get_summary_stats(crop_recommend_df.drop("label", axis = 1)))

#Test my fucntion against the Pandas mean() fucntion and compare if the results match.
print("My mean function: ", my_mean(test_df))
print("Pandas built in mean function: ", test_df.mean())
