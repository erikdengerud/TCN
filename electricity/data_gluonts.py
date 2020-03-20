# data_gluonts.py
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas

import matplotlib.pyplot as plt


print(f"Available datasets: {list(dataset_recipes.keys())}")

dataset = get_dataset("electricity", regenerate=False)



entry = next(iter(dataset.train))
train_series = to_pandas(entry)
train_series.plot()
plt.grid(which="both")
plt.legend(["train series"], loc="upper left")
plt.show()



entry = next(iter(dataset.test))
test_series = to_pandas(entry)
test_series.plot()
plt.axvline(train_series.index[-1], color='r') # end of train dataset
plt.grid(which="both")
plt.legend(["test series", "end of train series"], loc="upper left")
plt.show()


print(f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}")

print(f"Length of train series: {len(train_series)}")
print(f"Length of test series: {len(test_series)}")
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")