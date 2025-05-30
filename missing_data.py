"""
Module for detecting and dealing with missing data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import dataset
platform_data_df = pd.read_csv(filepath_or_buffer="datasets/clv_data.csv")
#print(platform_data_df.head())

def missing_summary(dataframe):
    """
    Function to return a missing data summary for a dataframe.
    """
    return pd.DataFrame({
        'Missing Count': dataframe.isnull().sum(),
        'Missing %': dataframe.isnull().mean() * 100,
        'Data Type': dataframe.dtypes
    }).sort_values(by='Missing %', ascending=False)

def plot_missing_data(dataframe):
    """
    Plot the missing data via a heatmap.
    """
    sns.heatmap(dataframe.isnull(), cbar=False)
    plt.title("Missing Data Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()


#Summary of missing data
#print(missing_summary(platform_data_df))
""" Shows the following missing data in the platform dataframe.
                  Missing Count  Missing % Data Type
age                        2446      48.92   float64
days_on_platform            141       2.82   float64
id                            0       0.00     int64
Unnamed: 0                    0       0.00     int64
gender                        0       0.00    object
income                        0       0.00     int64
city                          0       0.00    object
purchases                     0       0.00     int64
"""

#Show the heatmap of missing data.
plot_missing_data(platform_data_df)
