# run_electricity

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import sys
sys.path.append('')
sys.path.append('../../')
from datetime import date, timedelta

from model import TCN
from data import ElectricityDataSet
from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE
from utils.parser import parse, print_args


def train(epoch):
    tcn.train()
    total_loss = 0.0
    for i, d in enumerate(train_loader):
        x, y = d[0].to(device), d[1].to(device)
        optimizer.zero_grad()
        output = tcn(x)
        loss = criterion(output, y) / torch.abs(y).mean()
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        if i % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i*args.v_batch_size, length_dataset)
            writer.add_scalar('Loss/train', cur_loss, processed + length_dataset*epoch)
            if args.print:
                print(
                    (f"Train Epoch: {epoch:2d}"
                    f"[{processed:6d}/{length_dataset:6d}"
                    f"({100.*processed/length_dataset:.0f}%)]"
                    f"\tLearning rate: {args.lr:.4f}\tLoss: {cur_loss:.6f}"))
            total_loss = 0

def evaluate():
    tcn.eval()
    with torch.no_grad():
        for i, d in enumerate(test_loader):
            x, y = d[0].to(device), d[1].to(device)

            output = tcn(x)
            test_loss = criterion(output, y) / torch.abs(y).mean()

            predictions, real_values = tcn.rolling_prediction(x)
            real_values = real_values.cpu()
            predictions = predictions.cpu()

            mape = MAPE(real_values, predictions)
            smape = SMAPE(real_values, predictions)
            wape = WAPE(real_values, predictions)
            mae = MAE(real_values, predictions)
            rmse = RMSE(real_values, predictions)
            if args.print:
                print('Random batch of test set:')
                print('Test set: Loss: {:.6f}'.format(test_loss.item()))
                print('Test set: WAPE: {:.6f}'.format(wape))
                print('Test set: MAPE: {:.6f}'.format(mape))
                print('Test set: SMAPE: {:.6f}'.format(smape))
                print('Test set: MAE: {:.6f}'.format(mae))
                print('Test set: RMSE: {:.6f}'.format(rmse))
            return test_loss.item(), wape, mape, smape, mae, rmse

def evaluate_final():
    tcn.eval()
    with torch.no_grad():
        all_predictions = []
        all_real_values = []
        all_test_loss = []
        for i, data in enumerate(test_loader):
            x, y = data[0].to(device), data[1].to(device)

            predictions, real_values = tcn.rolling_prediction(x)
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

if __name__ == "__main__":
    args = parse()
    print_args(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    """ Dataset """
    print("Creating dataset.")
    # Lookback of the TCN
    look_back = 1 + 2 * (args.kernel_size -1) * 2**((args.num_layers+1)-1)
    print(f'Receptive field of the model is {look_back} time points.')
    look_back_timedelta = timedelta(hours=look_back)
    # Num rolling periods * Length of rolling period
    rolling_validation_length_days = timedelta(
        hours=args.num_rolling_periods*args.length_rolling)

    test_start = (
        date.fromisoformat(args.train_end) - 
        look_back_timedelta +
        timedelta(days=1)
        ).isoformat()
    test_end = (
        date.fromisoformat(args.train_end) + 
        rolling_validation_length_days + 
        timedelta(days=2)
        ).isoformat()
    print('Train dataset')
    train_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=args.train_start,
        end_date=args.train_end,
        h_batch=args.h_batch_size,
        include_time_covariates=args.time_covariates)
    print('Test dataset')
    test_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=args.time_covariates)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.v_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.v_batch_size, shuffle=True, num_workers=args.num_workers)
    length_dataset = train_dataset.__len__()

    """ TCN """
    tcn = TCN(
            num_layers=args.num_layers+1,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=args.kernel_size,
            residual_blocks_channel_size=[args.res_block_size]*args.num_layers + [1],
            bias=args.bias,
            dropout=args.dropout,
            stride=args.stride,
            dilations=None,
            leveledinit=args.leveledinit)
    tcn.to(device)
    print(
        f"""Number of learnable parameters : {
            sum(p.numel() for p in tcn.parameters() if p.requires_grad)}""")

    """ Training parameters"""
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(tcn.parameters(), lr=args.lr)

    """ Tensorboard """
    writer = SummaryWriter(log_dir=args.writer_path)

    """ Training """
    for ep in range(1, args.epochs+1):
        train(ep)
        tloss, wape, mape, smape, mae, rmse = evaluate()
        writer.add_scalar('Loss/test', tloss , ep)
        writer.add_scalar('wape', wape , ep)
        writer.add_scalar('mape', mape , ep)
        writer.add_scalar('smape', smape , ep)
        writer.add_scalar('mae', mae, ep)
        writer.add_scalar('rmse', rmse , ep)

    tloss, wape, mape, smape, mae, rmse = evaluate_final()
    print('Test set:')
    print('Loss: {:.6f}'.format(tloss))
    print('WAPE: {:.6f}'.format(wape))
    print('MAPE: {:.6f}'.format(mape))
    print('SMAPE: {:.6f}'.format(smape))
    print('MAE: {:.6f}'.format(mae))
    print('RMSE: {:.6f}'.format(rmse))

    writer.close()
    torch.save(tcn.state_dict(), args.model_save_path)
    print('Finished Training')

