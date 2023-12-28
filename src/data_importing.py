import pandas as pd


def import_data(file_path):
    df = pd.read_csv(file_path)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    return df
