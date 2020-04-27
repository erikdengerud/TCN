[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# TCN
An implementation of a Temporal Convolutional Network.

## Run experiments
Run files as modules using `python -m adding_problem.addtwo_run` or `python -m electricity\run_electricity.py`.

The run program takes in parsed arguments. Options are:

## Datasets
Currently there are two datasets we test on.

### The Adding Problem
This dataset has samples length T  and width 2. The first channel is random samples from a uniform[0,1] distribution. The second channel has two entries of 1 and the rest is zero. The Challenge is to add the two entries in the random channel where the zero one channel is one. This is the label of the sample.

### The Electricity Dataset
This dataset consists of time series of hourly electricity consumption of 370 houses <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>. Data are from the DeepGLO paper and can be 
downloaded usin the download_data.sh script in the data folder.

