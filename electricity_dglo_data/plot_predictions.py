# plot_predictions.py
"""
Plotting of the predictions from the models in models/.
Plotting the predictions for the test window 7x24h.
"""
from os import path
import torch
import pandas as pd
import matplotlib.pyplot as plt


def plot_predictions(model_path, data_loader, save_path):
    plot_look_back = False
    print(model_path)

    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # load data to plot
    iter_loader = iter(data_loader)
    x, y, idx = iter_loader.next()

    # predict using multi step and rolling predictions
    preds, actual = model.rolling_prediction(x, num_windows=7, tau=24)
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
    from data2 import ElectricityDataSet

    import torch
    from torch.utils.data import DataLoader

    from datetime import date, timedelta
    import glob

    """ Dataset """
    kernel_size = 7
    num_rolling_periods = 7
    length_rolling = 24
    train_end = "2014-12-16"
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
        "electricity_dglo_data/data/electricity.npy",
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=True,
        one_hot_id=False,
        receptive_field=look_back,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=v_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Get all models
    models = glob.glob("electricity_dglo_data/models/*.pt")
    print(models)
    for model_path in models:
        try:
            plot_predictions(
                model_path, test_loader, path.join("electricity_dglo_data/figures", ".".join([model_path, "pdf"]))
            )
        except Exception as e:
            print(e)
