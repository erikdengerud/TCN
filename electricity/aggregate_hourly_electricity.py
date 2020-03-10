import pandas as pd
import numpy as np


df = pd.read_csv('.\data\LD2011_2014.txt', sep=';', low_memory=False)
out = np.zeros(df.shape)
print('len of columns: ', len(df.columns))
for i, column in enumerate(df.columns):
    if i % 10 == 0:
        print(round(i/len(df.columns),3)*100)
    if i == 0:
        df[column] = pd.to_datetime(df[column])
    elif df[column].dtype == float:
        out[:,i] = df[column].values
    else:
        df[column] = df[column].str.replace(',','.')
        out[:,i] = df[column].values.astype(float)

new_df = pd.DataFrame(data=out)
new_df['date'] = df[df.columns[0]]

# Aggregate
times = pd.DatetimeIndex(new_df['date'])
df_hourly = new_df.groupby(
    [times.year, times.month, times.day, times.hour]
    ).sum()
print(df_hourly.head())


df_hourly.reset_index(drop=True, inplace=True)
print(df_hourly.head())
df_hourly.drop(df_hourly.columns[0], axis=1, inplace=True)
print(df_hourly.head())

# Add date col
date_rng = pd.date_range(start='1/1/2011', end='1/1/2015', freq='H')
print('Length of date_rng: ', len(date_rng))
df_hourly['date'] = date_rng
# Write aggregated df to file
df_hourly.to_csv(
    '.\data\LD2011_2014_hourly.txt'
    , sep=';', index=False)

'''
    # Cut off at start and end date
            new_df.index = new_df['date'] 
            new_df = new_df.loc[str(self.start_date):str(self.end_date)]
            new_df.reset_index(drop=True, inplace=True)

            # Aggregate
            times = pd.DatetimeIndex(new_df['date'])
            df_hourly = new_df.groupby(
                [times.year, times.month, times.day, times.hour]
                ).sum()
            print(df_hourly.head())


            df_hourly.reset_index(drop=True, inplace=True)
            df_hourly.drop(df_hourly.columns[0], axis=1, inplace=True)
            print(df_hourly.head())
            # Write aggregated df to file
            df_hourly.to_csv(
                '.\data\LD2011_2014_aggr_hourly_from_{}_to_{}.txt'.format(
                    self.start_date.isoformat(), self.end_date.isoformat())
                , sep=';', index=False)
'''