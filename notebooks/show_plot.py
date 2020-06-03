
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold


df = pd.read_csv("c:/Users/eriko/OneDrive - NTNU/ntnu/fag/prosjekt/TCN/revenue/data/processed_companies.csv", index_col=0).T
N = 10000
df["mean"] = df.mean(axis=1)

df = df.nlargest(N, "mean")

df = df.T
df = df / df.std(axis=0)

X = df.fillna(0).values

names = df.columns
node_position_model = manifold.TSNE(
    n_components=2, verbose=1)

embedding = node_position_model.fit_transform(X.T).T

df_plot = pd.DataFrame({"x": embedding[0], "y": embedding[1],  "name":names})


import plotly.graph_objects as go
import pandas as pd

fig = go.Figure(data=go.Scatter(x=df_plot["x"],
                                y=df_plot["y"],
                                mode='markers',
                                #marker=dict(size=df_plot["size"], color=df_plot["color"]),
                                text=df_plot["name"])) # hover text goes here

fig.update_layout(title='Embeddings TSNE')
fig.show()