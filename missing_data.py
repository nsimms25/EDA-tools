"""
Module for detecting and dealing with missing data.
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

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

