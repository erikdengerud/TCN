# Cluster Covariates for a TCN 
A clustering approach to capture global information as covariates for a Temporal Convolutional Network (TCN) in a multi time series setting. The cluster covariates are useful on some datasets, but not on others. They are useful on the datasets where the prototypes show different characteristics.

![Clustering of the time series in the Electricity dataset](https://github.com/erikdengerud/TCN/blob/master/Figures/electricity_best_clustering_edges.pdf)

![Prototype 0](https://github.com/erikdengerud/TCN/blob/master/Figures/electricity_best_cluster_0.pdf)
![Prototype 1](https://github.com/erikdengerud/TCN/blob/master/Figures/electricity_best_cluster_1.pdf)
![Prototype 8](https://github.com/erikdengerud/TCN/blob/master/Figures/electricity_best_cluster_8.pdf)

The prototypes are the black lines, and the filled area is the mid 80\% quantile.

## Folder structure

```bash
.
├── __init__.py
├── adding_problem
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── run_adding.py
│   └── README.md
├── clustering
│   ├── cluster.py
│   ├── clusters
│   │   └── ...
│   └── README.md
├── electricity
│   ├── __init__.py
│   ├── data
│   │   └── ...
│   ├── data.py
│   ├── figures
│   │   └── ...
│   ├── model.py
│   ├── run_electricity_2.py
│   ├── run_electricity_3.py
│   ├── run_electricity.py
│   ├── run_naive.py
│   ├── run_sarima.py
│   ├── run_sarimax.py
│   └── README.md
├── Figures
│   └── ...
├── notebooks
│   └── ...
├── prototypes
│   ├── create_future_prototypes_electricity.py
│   ├── create_future_prototypes.py
│   └── create_future_prototypes_revenue.py
├── representations
│   ├── representation_matrices
│   │   └── ...
│   ├── representations.py
│   ├── rep_sarima_revenue.npy
│   └── README.md
├── revenue
│   ├── __init__.py
│   ├── data
│   │   └── ...
│   ├── data_prep.py
│   ├── data.py
│   ├── figures
│   │   └── ...
│   ├── model.py
│   ├── run_naive.py
│   ├── run_revenue_2.py
│   ├── run_revenue_3.py
│   ├── run_revenue.py
│   ├── run_sarima.py
│   ├── run_sarimax.py
│   └── README.md
├── shapes
│   ├── __init__.py
│   ├── data.py
│   ├── figures
│   │   └── ...
│   ├── model.py
│   ├── run_shapes.py
│   └── README.md
├── similarities
│   ├── similarities.py
│   ├── similarity_matrices
│       └── ...
│   └── README.md
├── TCN
│   ├── __init__.py
│   ├── layers.py
│   └── tcn.py
└── utils
    └── ...
├── requirements.txt
├── README.md
```

## Run experiments

Each dataset has a `dataset\run_dataset.py` file that can be used to recreate the experiments. The run program takes in parsed arguments. Options are in `utils\parser`.

Each dataset has a dataset class in the `data.py` file in the speciific folder. The model
for the dataset is in the `model.py` file and it uses the TCN in the `TCN` folder. The `TCN`
uses the Dilated Causal 1D convolutional layer and the residual block in the `layers.py` file.
Training and evaluation of models happens in the `run_dataset.py` file in each folder.

## Datasets

### The Adding Problem
This dataset has samples length T  and width 2. The first channel is random samples from a uniform[0,1] distribution. The second channel has two entries of 1 and the rest is zero. The Challenge is to add the two entries in the random channel where the zero one channel is one. This is the label of the sample.

### The Electricity Dataset
This dataset consists of time series of hourly electricity consumption of 370 houses <https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014>. Data are from the DeepGLO paper and can be 
downloaded usin the `download_data.sh` script in the data folder.

### Revenue
A private dataset with total quarterly revenue for ~30,000 companies from 2007-2020.

