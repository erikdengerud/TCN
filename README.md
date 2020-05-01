# TCN
An implementation of a Temporal Convolutional Network.

## Folder structure

.
├── TCN
│   ├── __init__.py
│   ├── layers.py
│   └── tcn.py
├── adding_problem
│   ├── __init__.py
│   ├── README.md
│   ├── data.py
│   ├── model.py
│   └── run_adding.py
├── electricity
│   ├── __init__.py
│   ├── README.md
│   ├── data
│   │   └── download_data.sh
│   ├── data.py
│   ├── model.py
│   └── run_electricity.py
├── utils
│   ├── __init__.py
│   ├── metrics.py
│   ├── parser.py
│   ├── plot_predictions.py
│   ├── time.py
│   └── utils.py
├── __init__.py
├── README.md
├── requirements.txt
└── electricity_job.sh

## Run experiments
Run experiments like this `python electricity\run_electricity.py`.

The run program takes in parsed arguments. Options are in `utils\parser`.

Each dataset has a dataset class in the `data.py` file in the speciific folder. The model
for the dataset is in the `model.py` file and it uses the TCN in the `TCN` folder. The `TCN`
uses the Dilated Causal 1D convolutional layer and the residual block in the `layers.py` file.
Training and evaluation of models happens in the `run_dataset.py` file in each folder.
To run experiments on Idun one can use the script `electricity_job.sh`.

## Datasets

### The Adding Problem
This dataset has samples length T  and width 2. The first channel is random samples from a uniform[0,1] distribution. The second channel has two entries of 1 and the rest is zero. The Challenge is to add the two entries in the random channel where the zero one channel is one. This is the label of the sample.

### The Electricity Dataset
This dataset consists of time series of hourly electricity consumption of 370 houses <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>. Data are from the DeepGLO paper and can be 
downloaded usin the `download_data.sh` script in the data folder.

## Resources

https://arxiv.org/pdf/1901.10738.pdf Unsupervised representation learning.
https://arxiv.org/pdf/1603.04713.pdf Modeling time series similarity with siamese recurrent networks.
