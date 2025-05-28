import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import rankdata
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

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

def z_score_multi(dataframe, threshold=3):
    numeric_df = dataframe.select_dtypes(include='number')  #numeric columns
    outliers = {}

    for column in numeric_df.columns:
        data = pd.to_numeric(dataframe[column], errors='coerce').dropna()
        column_mean = data.mean()
        column_stddev = data.std()

        z_scores_col = (data - column_mean) / column_stddev

        outliers[column] = dataframe.loc[abs(z_scores_col) > threshold, column]

    return outliers

def iqr_mulit(dataframe):
    numeric_df = dataframe.select_dtypes(include='number')  #numeric columns
    outliers = {}

    for column in numeric_df.columns:
        data = pd.to_numeric(dataframe[column], errors='coerce').dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[column] = dataframe.loc[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound), column]

    return outliers

def add_outlier_mask_column(dataframe, outliers):
    result_dataframe = dataframe.copy()

    for column, outlier_series in outliers.items():
        outlier_mask = dataframe.index.isin(outlier_series.index)
        result_dataframe[f"{column}_outlier"] = outlier_mask
    
    return result_dataframe

def get_corr_matrix(dataframe):
    matrix = dataframe.corr(numeric_only = True)

    return matrix

#Time to create a Correlation Matrix by hand without using the Pandas builtin function.
def my_corr_matrix(dataframe):
    numeric_df = dataframe.select_dtypes(include='number')
    columns = numeric_df.columns
    num = len(columns)

    corr_df = pd.DataFrame(np.zeros((num, num)), columns=columns, index=columns)

    for i in range(num):
        for j in range(num):
            x = numeric_df[columns[i]]
            y = numeric_df[columns[j]]

            x_mean = x.mean()
            y_mean = y.mean()

            sum_deviations = sum((x - x_mean) * (y - y_mean))
            product_std_deviations = np.sqrt(sum((x - x_mean) ** 2)) * np.sqrt(sum((y - y_mean) ** 2))

            correlation = sum_deviations / product_std_deviations if product_std_deviations != 0 else np.nan
            corr_df.iloc[i, j] = correlation
    
    return corr_df.round(3)

def visualize_corr_matrix(corr_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0, 
        square=True, 
        linewidths=0.5, 
        fmt=".2f"
    )
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def get_spearman_corr_matrix(dataframe):
    matrix = dataframe.corr(numeric_only = True, method = "spearman", )

    return matrix

def my_spearman_corr_matrix(dataframe):
    numeric = dataframe.select_dtypes(include='number')
    columns = numeric.columns
    nums = len(columns)
    corr_df = pd.DataFrame(np.zeros((nums, nums)), columns=columns, index=columns)

    for i in range(nums):
        for j in range(nums):
            x = rankdata(numeric.iloc[:, i])
            y = rankdata(numeric.iloc[:, j])

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))

            corr = numerator / denominator if denominator != 0 else np.nan
            corr_df.iloc[i, j] = corr
        
    return corr_df.round(3)

def get_kendall_corr_matrix(dataframe):
    matrix = dataframe.corr(numeric_only = True, method = "kendall", )

    return matrix

def get_kendall_tau(x, y):
    n = len(x)
    concordant = discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            a = x[i] - x[j]
            b = y[i] - y[j]
            prod = a * b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # Ignore ties

    total_pairs = n * (n - 1) / 2
    tau = (concordant - discordant) / total_pairs if total_pairs != 0 else np.nan
    return tau

def my_kendall_corr_matrix(df):
    numeric_df = df.select_dtypes(include='number')
    cols = numeric_df.columns
    n = len(cols)
    corr_df = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

    for i in range(n):
        for j in range(n):
            x = numeric_df.iloc[:, i].to_numpy()
            y = numeric_df.iloc[:, j].to_numpy()
            corr_df.iloc[i, j] = get_kendall_tau(x, y)

    return corr_df.round(2)

def my_pairplot(dataframe):
    sns.pairplot(dataframe, hue=None, corner=True, diag_kind="kde", height=1, aspect=1)
    plt.tight_layout()
    plt.suptitle("Pairplot", y=0.9)
    plt.show()

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

#This is for mulit column z-scores:
#Now the z_scores can be called base on the column needed. 
z_scores = z_score_multi(crop_recommend_df, threshold=4)
#Uncomment to show z_scores for  "ph" column.
#print(z_scores["ph"])

#This is for mulit column IQR:
#Now the IQR can be called base on the column needed. 
iqr_data = iqr_mulit(crop_recommend_df)
#Uncomment to show z_scores for  "ph" column.
#print(iqr_data["temperature"])

#Use mask function to return a copy of the Dataframe with outlier mask column.
new_df = add_outlier_mask_column(crop_recommend_df, z_scores)
#print(new_df.head())

#Compare the builtin correlation matrix from Pandas to a built from scratch one.
#print(get_corr_matrix(crop_recommend_df))
#print(my_corr_matrix(crop_recommend_df))

#Create a heatmap of the correlation matrix.
#visualize_corr_matrix(get_corr_matrix(crop_recommend_df))

#Compare the builtin correlation matrix from Pandas to a built from scratch one.
#print(get_spearman_corr_matrix(crop_recommend_df))
#print(my_spearman_corr_matrix(crop_recommend_df))

#Compare the builtin correlation matrix from Pandas to a built from scratch one.
#print(get_kendall_corr_matrix(crop_recommend_df))
#print(my_kendall_corr_matrix(crop_recommend_df)) #This will take a while due to df size.

#Create a pairplot from seaborn and matplotlib.
my_pairplot(crop_recommend_df)
