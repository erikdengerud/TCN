# The Adding Problem

This is a dataset used to see if the TCN was implemented correctly. The task is to add two random numbers. This can be challenging for RNNs and LSTMs even though they theoretically have infinite memory.

## Data

The data has two channels of length 'seq_length' . The first is randomly genereated numbers in (0,1) and the second is zeros eccept two entries which are one. The target is the sum of the random numbers where the entries are one.

## Model

The TCN is adjusted to predict a single number by adding a linear layer on top of the TCN.