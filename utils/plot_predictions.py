# plot_predictions.py
"""
Plot the predictions of a model on a dataset.
This function is made for creating plots to display in tenorboard and therefore
saves it to a buffer.
"""
import sys

sys.path.append("")
sys.path.append("./")
sys.path.append(".")
sys.path.append("../../")

import io
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader


def plot_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device,
    num_windows: int = 7,
    tau: int = 24,
    num_to_plot=4,
) -> plt.Figure:
    """
    Plotting predictions by the model on the test set.
    """
    # Load data to plot. The dataloader should have shuffle=False to get the same time
    # series each time.
    iter_loader = iter(data_loader)
    x, y, idx = iter_loader.next()
    x = x.to(device)
    y = y.to(device)
    idx = idx.to(device)
    x = x[:num_to_plot]
    y = y[:num_to_plot]
    idx = idx[:num_to_plot]
    # Predict using multi step and rolling predictions
    preds, actual = model.rolling_prediction(x, emb_id=idx, num_windows=7, tau=24)
    # plot n series at a time. Real and predicted values
    preds, actual = preds.detach().cpu().numpy(), actual.detach().cpu().numpy()
    num_series = preds.shape[0]
    dfs = []
    for i in range(len(preds)):
        df = pd.DataFrame(data=[preds[i], actual[i]])
        df = df.T
        df.columns = ["predicted", "target"]
        dfs.append(df)
    fig, axes = plt.subplots(nrows=num_series, ncols=1, sharex=True)

    vlines = [24 * i for i in range(7)]
    for i in range(len(dfs)):
        df = dfs[i]
        dfs[i].plot(ax=axes[i], legend=False)
        axes[i].set_ylabel(idx[i].item(), rotation=0)
        for vline in vlines:
            axes[i].axvline(x=vline, color="r")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    plt.subplots_adjust()  # right=0.85)
    plt.setp(axes, xticks=vlines, xticklabels=vlines)

    return fig


if __name__ == "__main__":
    from electricity_dglo_data.data import ElectricityDataSet
    from electricity_dglo_data.model import TCN

    import torch
    from torch.utils.data import DataLoader

    from datetime import date, timedelta
    from torch.utils.tensorboard import SummaryWriter

    """ Dataset """
    kernel_size = 7
    num_rolling_periods = 7
    length_rolling = 24
    num_workers = 0
    save_path = "plot_predtictions.pdf"

    print("Creating dataset.")
    test_dataset = ElectricityDataSet(
        "electricity/data/electricity.npy",
        start_date="2012-01-01",
        end_date="2013-01-01",
        h_batch=0,
        include_time_covariates=False,
        one_hot_id=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=4, shuffle=False, num_workers=0,
    )

    """ Initialize model """
    tcn = TCN(
        num_layers=5,
        in_channels=1,
        out_channels=1,
        kernel_size=7,
        residual_blocks_channel_size=[16] * 5,
        bias=True,
        leveledinit=False,
    )
    print(
        f"""Number of learnable parameters : {
            sum(p.numel() for p in tcn.parameters() if p.requires_grad)}"""
    )

    """ Tensorboard """
    writer = SummaryWriter(log_dir="test")

    """ Training """
    fig = plot_predictions(tcn, test_loader)

    writer.add_figure("test", fig)

    writer.close()
    print("Finished Training")
