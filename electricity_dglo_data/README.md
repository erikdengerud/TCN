# Electricity

This folder contains the experiments on the electricity dataset. 

## Data

The data comes from the authors of the [DeepGLO](https://arxiv.org/pdf/1905.03806.pdf) paper. It can be downloaded using the script `download_data.sh` in the `data` folder.

The electricity dataset is part of the UCI archive [Link](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014).
It contains the electricity consumption for 370 clients/houses in the period from 2011 to 2014.
The data is in the form of averages over 15 minutes intervals, but it is used as 1 hour intervals in DeepGLO and other papers. 

### Statisitics and Properties
The dataset consists of 370 rows, one for each client/house.  It has no missing values. New clients after 2011 have their consumption set to zero in the period before they became clients. Values are in kw and we convert to kwh in our processed dataset.
We also have the date as a column and we create our time covariates from this column. We create 7 time covariates: minute of
the hour, hour of the day, day of the week, day of the month, day of the year, month of the year and week of the
year. The time covariates are floats in [-0.5, 0.5]. Looking at it now it's unclear why you would use minute of the hour with hourly data. This is however what they say they do in DeepGLO.
In DeepGLO they use the data from 2012-2014. This is because many of the entries from 2011 contains many zeros. We therefore use the same in our default dataset. 

The length of the time series is just under 2600 time points. The time series looks noisy and show less signs of seasonality throughout a day than we would expect. They do however show some of the same patterns. We also expect them to be highly correlated, but this isn't too clear when we plot them.
![examples](figures/example_ts.pdf)

## Model

## Results

## Discussion