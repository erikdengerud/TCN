# plot_predictions.py
"""
Plotting of the predictions from the models in models/.
Plotting the predictions for the test window 7x24h.
"""
from os import path
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(model_path, data_loader, save_path, model_class):
    plot_look_back = False
    print(model_path)
    # load model
    if "_tc" in model_path or "random" in model_path:
        in_channels = 8
    else:
        in_channels = 1
    print(f"In channels: {in_channels}")
    model = model_class(
        in_channels=in_channels,
        num_layers=6,
        out_channels=1,
        residual_blocks_channel_size=[32] * 5 + [1],
        kernel_size=7,
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    # load data to plot
    iter_loader = iter(data_loader)
    x, y, idx = iter_loader.next()

    # predict using multi step and rolling predictions
    preds, actual = model.rolling_prediction(x, y, num_windows=7, tau=24)
    # plot n series at a time. Real and predicted values
    preds, actual = preds.detach().numpy(), actual.detach().numpy()
    num_series = preds.shape[0]
    print(f"num series = {num_series}")
    dfs = []
    for i in range(len(preds)):
        df = pd.DataFrame(data=[preds[i], actual[i]])
        df = df.T
        df.columns = ["predicted", "target"]

        dfs.append(df)  # , columns=['predicted', 'target']))
    fig, axes = plt.subplots(nrows=num_series, ncols=1, sharex=True)

    if plot_look_back:
        y = y.detach().numpy()
        vlines = [model.lookback + 24 * i for i in range(7)]
        for i in range(len(dfs)):
            end = y.shape[2]
            axes[i].set_xlim(0, end)
            axes[i].plot(range(end), y[i].T, color="b", label="Target")
            pred_range = [model.lookback + i for i in range(len(preds[i]))]
            axes[i].plot(pred_range, preds[i], color="g", label="Predictions")
            axes[i].set_ylabel(idx[i].item())
            for vline in vlines:
                axes[i].axvline(x=vline)
            axes[i].legend()
        fig.suptitle(model_path)
    else:
        y = y.detach().numpy()
        vlines = [24 * i for i in range(7)]
        for i in range(len(dfs)):
            df = dfs[i]
            # Add values before rolling predictions
            dfs[i].plot(ax=axes[i], logy=False, legend=False)
            axes[i].set_ylabel(idx[i].item(), rotation=0)
            for vline in vlines:
                axes[i].axvline(x=vline)
        fig.suptitle(model_path)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right")
    plt.subplots_adjust()  # right=0.85)
    plt.setp(axes, xticks=vlines, xticklabels=vlines)

    # save plots
    plt.savefig(save_path)


if __name__ == "__main__":
    from data import ElectricityDataSet
    from model import TCN

    import torch
    from torch.utils.data import DataLoader

    from datetime import date, timedelta
    import glob

    """ Dataset """
    kernel_size = 7
    num_rolling_periods = 7
    length_rolling = 24
    train_end = "2014-12-17"
    time_covariates = True
    one_hot_id = False
    v_batch_size = 5
    num_workers = 0
    save_path = "plot_predtictions.pdf"

    print("Creating dataset.")
    # Lookback of the TCN
    look_back = 1 + 2 * (kernel_size - 1) * 2 ** ((5 + 1) - 1)
    print(f"Receptive field of the model is {look_back} time points.")
    look_back_timedelta = timedelta(hours=look_back)
    # Num rolling periods * Length of rolling period
    rolling_validation_length_days = timedelta(
        hours=num_rolling_periods * length_rolling
    )

    test_start = (
        date.fromisoformat(train_end) - look_back_timedelta + timedelta(days=1)
    ).isoformat()
    test_end = (
        date.fromisoformat(train_end)
        + rolling_validation_length_days
        + timedelta(days=2)
    ).isoformat()

    print("Test dataset")
    test_dataset = ElectricityDataSet(
        # "electricity/data/random_dataset.txt",
        "electricity/data/LD2011_2014_hourly.txt",
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=False,
        one_hot_id=one_hot_id,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=v_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataset_tc = ElectricityDataSet(
        "electricity/data/LD2011_2014_hourly.txt",
        # "electricity/data/random_dataset.txt",
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=True,
        one_hot_id=one_hot_id,
    )
    test_loader_tc = DataLoader(
        dataset=test_dataset_tc,
        batch_size=v_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Get all models
    models = glob.glob("electricity/models/*.pt")
    print(models)
    for model_path in models:
        if "_tc" in model_path:
            plot_predictions(
                model_path, test_loader_tc, ".".join([model_path, "pdf"]), TCN
            )
            print("tc loader")
        elif "random" in model_path:
            plot_predictions(
                model_path, test_loader_tc, ".".join([model_path, "pdf"]), TCN
            )
            print("tc loader")

        else:
            plot_predictions(
                model_path, test_loader, ".".join([model_path, "pdf"]), TCN
            )
            print("not tc loader")
