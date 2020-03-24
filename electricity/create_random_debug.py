# create_random_debug.py
"""
Create a random dataset to debug our model.
"""

import torch
import pandas as pd
import numpy as np

def create(destination_path):
    """ Date column and 370 columns """
    dates = pd.date_range(start='2012-01-01', end='2015-01-01', freq='H')
    print(dates)
    print(len(dates))
    mat = np.random.rand(370, len(dates)).T
    df = pd.DataFrame(mat)
    df['date'] = dates
    print(df.head())
    df.to_csv(destination_path, sep=';', index=False)

if __name__ == "__main__":
    create(destination_path='electricity/data/random_dataset.txt')