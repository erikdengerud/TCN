# run_electricity

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
import argparse
import sys
sys.path.append('')
sys.path.append('../../')

from model import TCN
from data import ElectricityDataSet
from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE

def parse():
    parser = argparse.ArgumentParser(description='Adding Problem')
    parser.add_argument(
        '--train_start', type=str, default='2012-01-01', metavar='train_start')
    parser.add_argument(
        '--train_end', type=str, default='2014-08-01', metavar='train_end')
    parser.add_argument(
        '--test_start', type=str, default='2014-08-01', metavar='test_start')
    parser.add_argument(
        '--test_end', type=str, default='2014-10-01', metavar='test_end')
    parser.add_argument(
        '--v_batch_size', type=int, default=32, metavar='v_batch_size')
    parser.add_argument(
        '--h_batch_size', type=int, default=256, metavar='h_batch_size')
    parser.add_argument(
        '--num_layers', type=int, default=5, metavar='num_layers')
    parser.add_argument(
        '--in_channels', type=int, default=8, metavar='in_channels')
    parser.add_argument(
        '--out_channels', type=int, default=1, metavar='out_channels')
    parser.add_argument(
        '--kernel_size', type=int, default=7, metavar='kernel_size')
    parser.add_argument(
        '--res_block_size', type=int, default=32, metavar='res_block_size')
    parser.add_argument(
        '--bias', type=bool, default=True, metavar='bias')
    parser.add_argument(
        '--dropout', type=float, default=0.0, metavar='dropout')
    parser.add_argument(
        '--stride', type=int, default=1, metavar='stride')
    parser.add_argument(
        '--leveledinit', type=bool, default=False, metavar='leveledinit')
    parser.add_argument(
        '--model_save_path', type=str, default='electricity/models/tcn_electricity.pt', 
        metavar='model_save_path')
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='epochs')
    parser.add_argument(
        '--lr', type=float, default=5e-4, metavar='lr')
    parser.add_argument(
        '--clip', type=bool, default=False, metavar='clip')
    parser.add_argument(
        '--log_interval', type=int, default=5, metavar='log_interval')
    parser.add_argument(
        '--writer_path', type=str, default='electricity/runs/electricity_1', 
        metavar='writer_path')
    parser.add_argument(
        '--print', type=bool, default=False, metavar='print')
    parser.add_argument(
        '--num_workers', type=int, default=0, metavar='num_workers')
    args = parser.parse_args()
    return args

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
            writer.add_scalar('training_loss', cur_loss, processed + length_dataset*epoch)
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

        #if args.print:
        #    print('Loss: {:.6f}'.format(test_loss))
        #    print('WAPE: {:.6f}'.format(wape))
        #    print('MAPE: {:.6f}'.format(mape))
        #    print('SMAPE: {:.6f}'.format(smape))
        return test_loss, wape, mape, smape, mae, rmse

if __name__ == "__main__":
    args = parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    """ Dataset """
    print("Creating dataset.")
    train_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=args.train_start,
        end_date=args.train_end,
        h_batch=args.h_batch_size,
        include_time_covariates=True)
    test_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=args.test_start,
        end_date=args.test_end,
        h_batch=0,
        include_time_covariates=True)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.v_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.v_batch_size, shuffle=True, num_workers=args.num_workers)
    length_dataset = train_dataset.__len__()

    """ TCN """
    tcn = TCN(
            num_layers=args.num_layers,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=args.kernel_size,
            residual_blocks_channel_size=[args.res_block_size]*args.num_layers,
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
    writer = SummaryWriter(args.writer_path)

    """ Training """
    for ep in range(1, args.epochs+1):
        train(ep)
        tloss, wape, mape, smape, mae, rmse = evaluate()
        writer.add_scalar('test_loss', tloss , ep)
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

