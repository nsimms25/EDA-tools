import pandas as pd
import numpy as np
import scipy.stats

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
    """
    This is my own creation for mean, this will not be useful over pandas or numpy.
    This is for statistics practice in python.

    Args:
        dataframe (DataFrame): Pandas DataFrame 
    
    Returns:
        Mean (float): mean statistic for the dataframe input.
    """
    row_count = dataframe.shape[0]
    func_count = my_count(dataframe)
    func_sum = my_sum(dataframe)

    mean = func_sum / func_count

    return mean

def my_std_stat(dataframe):
    """
    This is my own creation for std, this will not be useful over pandas or numpy.
    This is for statistics practice in python.

    Args:
        dataframe (DataFrame): Pandas DataFrame 
    
    Returns:
        std (float): std statistic for the dataframe input.
    """
    row_count = dataframe.shape[0]
    func_count = my_count(dataframe)
    func_mean = my_mean(dataframe)

    df = dataframe.iloc[:, 0]
    values = pd.to_numeric(df, errors='coerce').dropna()

    sq_diffs = sum((x - func_mean) ** 2 for x in values)
    variance = sq_diffs / (func_count - 1)

    return variance ** 0.5

def z_score_series(dataframe, threshold=3):
    """
    Find the zscore and outliers for a pandas series, hence series in function name.
    """

    data = dataframe.iloc[:, 0]
    values = pd.to_numeric(data, errors='coerce').dropna()

    mean = data.mean()
    stddev = data.std()

    z_scores = (data - mean) / stddev

    outliers = data[abs(z_scores) > threshold]

    return outliers

def iqr_series(dataframe):
    data = pd.to_numeric(dataframe.iloc[:, 0], errors='coerce').dropna()

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return outliers

#TODO: Do these for multiple column df..
#1. Z-score
#2. Outlier Detection
#    IQR based


#TODO: 
#3. Correlation Matrix with Visual

#TODO:
#4. Pairplot & Distributions
#    Use sns.pairplot()
#    sns.histplot() or sns.kdeplot() for each numerical feature

#Nice-to-Have Extras
#    Profiling report export (like pandas-profiling or ydata-profiling)
#    Categorical variable summaries (value counts, mode, entropy)
#    Save plots to files or PDF report

#Import dataset
crop_recommend_df = pd.read_csv(filepath_or_buffer="datasets/Crop_recommendation.csv")
#print(df.head())

test_df = crop_recommend_df.drop(["label"], axis = 1)
#print(test_df.head())

def get_summary_stats(dataframe):
    return dataframe.describe().T.assign(
        missing_pct=dataframe.isnull().mean() * 100,
        skewness=dataframe.skew(),
        kurtosis=dataframe.kurt()
    )

#Test my function against the Pandas mean() function and compare if the results match.
#print("My mean function: \n", my_mean(test_df))
#print("Pandas built in mean function: \n", test_df.mean())
#Results match the builtin Pandas function.

