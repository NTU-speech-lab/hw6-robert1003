import pandas as pd
import numpy as np

def read_label(path, categories):
    df = pd.read_csv(path)
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(categories)
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()

    return df, label_name
