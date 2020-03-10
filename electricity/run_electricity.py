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
from utils.metrics import WAPE, MAPE, SMAPE

def parse():
    parser = argparse.ArgumentParser(description='Adding Problem')
    parser.add_argument(
        '--train_start', type=str, default='2012-01-01', metavar='train_start')
    parser.add_argument(
        '--train_end', type=str, default='2014-06-01', metavar='train_end')
    parser.add_argument(
        '--test_start', type=str, default='2014-06-15', metavar='test_start')
    parser.add_argument(
        '--test_end', type=str, default='2014-12-31', metavar='test_end')
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
        loss = criterion(output, y)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item()

        if i % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i*args.v_batch_size, length_dataset)
            writer.add_scalar('training_loss', cur_loss, processed)
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
        for data in test_loader:
            x, y = data[0].to(device), data[1].to(device)
            output = tcn(x)
            test_loss = test_loss = criterion(output, y)
            y = y.cpu()
            output = output.cpu()
            mape = MAPE(y, output)
            wape = WAPE(y, output)
            smape = SMAPE(y, output)
            if args.print:
                print('Test set: Loss: {:.6f}'.format(test_loss.item()))
                print('Test set: WAPE: {:.6f}'.format(wape))
                print('Test set: MAPE: {:.6f}'.format(mape))
                print('Test set: SMAPE: {:.6f}'.format(smape))
            return test_loss.item(), wape, mape, smape

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
        h_batch=args.h_batch_size,
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
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tcn.parameters(), lr=args.lr)

    """ Tensorboard """
    writer = SummaryWriter(args.writer_path)

    """ Training """
    for ep in range(1, args.epochs+1):
        train(ep)
        tloss, wape, mape, smape = evaluate()
        writer.add_scalar('test_loss', tloss , ep)
        writer.add_scalar('wape', wape , ep)
        writer.add_scalar('mape', mape , ep)
        writer.add_scalar('smape', smape , ep)

    writer.close()
    torch.save(tcn.state_dict(), args.model_save_path)
    print('Finished Training')

