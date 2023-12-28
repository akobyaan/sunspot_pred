import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def data_information(df):
    print(df.shape)
    print(df.head())
    print(df.info())
    print(df.isna().sum())
    df.replace(-1, np.nan, inplace=True)
    df.interpolate(inplace=True)
    df.fillna(df.mean(), inplace=True)
    print(f"Correlation matrix: {df.corr()}")
    sns.pairplot(df)
    plt.suptitle('Pair Plot of Multiple Features', y=1.02)
    plt.show()


def ind_dep_columns(df):
    X = df.drop(columns='Number of Sunspots')
    y = df['Number of Sunspots']
    # After correlation
    df.drop(['Month', 'Day', 'Observations', 'Indicator'], axis=1, inplace=True)
    return X, y
