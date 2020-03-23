# plot_predictions.py
"""
Plotting of the predictions from the models in models/.
Plotting the predictions for the test window 7x24h.
"""
from os import path
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(model_path, data_loader, save_path, model_class):
    print(model_path)
    # load model
    if '_tc' in model_path:
        in_channels=8
    else:
        in_channels=1
    print(f"In channels: {in_channels}")
    model = model_class(
        in_channels=in_channels, 
        num_layers=6, 
        out_channels=1, 
        residual_blocks_channel_size=[32]*5 + [1],
        kernel_size=7)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # load data to plot
    iter_loader = iter(data_loader)
    x, y = iter_loader.next()
    print(x.shape)
    print(y.shape)
    print(y[0,:,model.lookback:model.lookback+6])

    # predict using multi step and rolling predictions
    preds, actual = model.rolling_prediction(x, y, num_windows=1, tau=6)
    print(preds.shape)
    print(actual.shape)
    # plot n series at a time. Real and predicted values
    preds, actual = preds.detach().numpy(), actual.detach().numpy()
    print(preds.shape)
    print(actual.shape)
    num_series = preds.shape[0]
    print(f'num series = {num_series}')
    dfs = []
    print(len(preds))
    for i in range(len(preds)):
        df = pd.DataFrame(data=[preds[i], actual[i]])
        df = df.T
        df.columns = ['predicted', 'target']
        
        dfs.append(df)#, columns=['predicted', 'target']))
        print(dfs[i].head())
    fig, axes = plt.subplots(nrows=num_series, ncols=1)
    print(len(dfs))
    print(dfs[3])
    for i in range(len(dfs)):
        dfs[i].plot(ax=axes[i])
        # add red lines every 24 hours
    plt.show()
    
    
    print(preds)
    print(actual)
    print(preds.shape)
    print(actual.shape)
    # save plots
    plt.savefig(save_path)

    print(f"loaded {model_path}")

'''
def evaluate_final():
    tcn.eval()
    with torch.no_grad():
        all_predictions = []
        all_real_values = []
        all_test_loss = []
        for i, data in enumerate(test_loader):
            x, y = data[0].to(device), data[1].to(device)

            predictions, real_values = tcn.rolling_prediction(x, y)
            all_predictions.append(predictions)
            all_real_values.append(real_values)
            
            output = tcn(x)
            test_loss = criterion(output, y) / torch.abs(y).mean()
            all_test_loss.append(test_loss.item())

        predictions_tensor = torch.cat(all_predictions, 0)
        real_values_tensor = torch.cat(all_real_values, 0)

        predictions_tensor = predictions_tensor.cpu()
        real_values_tensor = real_values_tensor.cpu()

        mape = MAPE(real_values_tensor, predictions_tensor)
        smape = SMAPE(real_values_tensor, predictions_tensor)
        wape = WAPE(real_values_tensor, predictions_tensor)
        test_loss = np.sum(all_test_loss)
        mae = MAE(real_values_tensor, predictions_tensor)
        rmse = RMSE(real_values_tensor, predictions_tensor)

        return test_loss, wape, mape, smape, mae, rmse
'''

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
    train_end = '2014-12-17'
    time_covariates = True
    one_hot_id = False
    v_batch_size = 4
    num_workers = 0
    save_path = 'plot_predtictions.pdf'

    print("Creating dataset.")
    # Lookback of the TCN
    look_back = 1 + 2 * (kernel_size - 1) * 2**((5+1)-1)
    print(f'Receptive field of the model is {look_back} time points.')
    look_back_timedelta = timedelta(hours=look_back)
    # Num rolling periods * Length of rolling period
    rolling_validation_length_days = timedelta(
        hours=num_rolling_periods * length_rolling)

    test_start = (
        date.fromisoformat(train_end) - 
        look_back_timedelta +
        timedelta(days=1)
        ).isoformat()
    test_end = (
        date.fromisoformat(train_end) + 
        rolling_validation_length_days + 
        timedelta(days=2)
        ).isoformat()

    print('Test dataset')
    test_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=False,
        one_hot_id=one_hot_id)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=v_batch_size, shuffle=True, num_workers=num_workers)
    test_dataset_tc = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=True,
        one_hot_id=one_hot_id)
    test_loader_tc = DataLoader(
        dataset=test_dataset_tc, batch_size=v_batch_size, shuffle=True, num_workers=num_workers)

    tc_iter = iter(test_loader_tc)
    x, y = tc_iter.next()
    print(x.shape)
    print(y.shape)
    no_tc_iter = iter(test_loader)
    x, y = no_tc_iter.next()
    print(x.shape)
    print(y.shape)

    # Get all models
    models = glob.glob('electricity/models/*.pt')
    print(models)
    for model_path in models:
        if '_tc' in model_path:
            plot_predictions(model_path, test_loader_tc, save_path, TCN)
            print('tc loader')
        else:
            plot_predictions(model_path, test_loader, save_path, TCN)
            print('not tc loader')
        
        
        # predict

    
    # plot predictions vs actual

    # save 
    