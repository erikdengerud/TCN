# Shape dataset

This is  a dataset with 3 shapes, 3 frequencies and 3 noise setting.
We use the dataset to test embeddings and use a 2 dimensional embedding space.

## Data

The dataset contains N time series of length t.

### Shapes:
* Sawtooth
* Rectangular
* Sine

The shapes are made using the functions in `scipy.signal`.

### Noise:
* No noise
* IID
* Heavily correlated long effects

The IID we use is N(0,1).

### Frequency:
* 1
* 3
* 9

Frequencies can be specified in the `scipy.signal` functions.
