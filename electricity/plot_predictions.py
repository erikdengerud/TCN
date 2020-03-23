# plot_predictions.py
"""
Plotting of the predictions from the models in models/.
Plotting the predictions for the test window 7x24h.
"""

def plot_predictions(model, dataset, save_path):
    # load model

    # load data

    # predict using multi step and rolling predictions

    # plot n series at a time. Real and predicted values

    # save plots
    pass


if __name__ == "__main__":
    from data import ElectricityDataSet

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
    v_batch_size = 32
    num_workers = 0
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
        include_time_covariates=time_covariates,
        one_hot_id=one_hot_id)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=v_batch_size, shuffle=True, num_workers=num_workers)

    # Get all models
    models = glob.glob('electricity/models/*.pt')
    print(models)
    # for model in models
        #model = TheModelClass(*args, **kwargs)
        #model.load_state_dict(torch.load(PATH))
        #model.eval()
        # predict

    
    # plot predictions vs actual

    # save 
    