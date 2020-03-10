# addtwo_run-py
"""
Train and test a TCN on the add two dataset.
Trying to reproduce https://arxiv.org/abs/1803.01271.
"""
print('Importing modules')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import argparse
import sys
sys.path.append('')
sys.path.append("../../")

from data import AddTwoDataSet
from model import TCN
print('modules imported')

def parse():
    parser = argparse.ArgumentParser(description='Adding Problem')
    parser.add_argument(
        '--N_train', type=int, default=50000, metavar='N_train')
    parser.add_argument(
        '--N_test', type=int, default=1000, metavar='N_test')
    parser.add_argument(
        '--seq_length', type=int, default=200, metavar='seq_length')
    parser.add_argument(
        '--batch_size', type=int, default=32, metavar='batch_size')
    parser.add_argument(
        '--num_layers', type=int, default=8, metavar='num_layers')
    parser.add_argument(
        '--in_channels', type=int, default=2, metavar='in_channels')
    parser.add_argument(
        '--out_channels', type=int, default=1, metavar='out_channels')
    parser.add_argument(
        '--kernel_size', type=int, default=7, metavar='kernel_size')
    parser.add_argument(
        '--res_block_size', type=int, default=30, metavar='res_block_size')
    parser.add_argument(
        '--bias', type=bool, default=True, metavar='bias')
    parser.add_argument(
        '--dropout', type=float, default=0.0, metavar='dropout')
    parser.add_argument(
        '--stride', type=int, default=1, metavar='stride')
    parser.add_argument(
        '--leveledinit', type=bool, default=False, metavar='leveledinit')
    parser.add_argument(
        '--model_save_path', type=str, default='adding_problem/models/tcn_addtwo.pt', 
        metavar='model_save_path')
    parser.add_argument(
        '--epochs', type=int, default=10, metavar='epochs')
    parser.add_argument(
        '--lr', type=float, default=2e-3, metavar='lr')
    parser.add_argument(
        '--clip', type=bool, default=False, metavar='clip')
    parser.add_argument(
        '--log_interval', type=int, default=100, metavar='log_interval')
    parser.add_argument(
        '--writer_path', type=str, default='adding_problem/sruns/add_two1', 
        metavar='writer_path')
    parser.add_argument(
        '--print', type=bool, default=False, metavar='print')
    parser.add_argument(
        '--num_workers', type=int, default=0, metavar='num_workers')
    args = parser.parse_args()
    return args

def run():
    torch.manual_seed(1729)
        
    """ Setup """
    args = parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    """ Dataset """
    train_dataset = AddTwoDataSet(N=args.N_train, seq_length=args.seq_length)
    test_dataset = AddTwoDataSet(N=args.N_test, seq_length=args.seq_length)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    """ TCN """
    tcn = TCN(
            num_layers=args.num_layers,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=args.kernel_size,
            residual_blocks_channel_size=[args.res_block_size] * args.num_layers,
            bias=args.bias,
            dropout=args.dropout,
            stride=args.stride,
            dilations=None,
            leveledinit=args.leveledinit)
    tcn.to(device)
    if args.print:
        print(
            f"""Number of learnable parameters : {
                sum(p.numel() for p in tcn.parameters() if p.requires_grad)}""")

    """ Training parameters"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tcn.parameters(), lr=args.lr)

    """ Tensorboard """
    writer = SummaryWriter(args.writer_path)

    for ep in range(1, args.epochs+1):
        """ TRAIN """
        tcn.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            output = tcn(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()

            if i % args.log_interval == 0:
                cur_loss = total_loss / args.log_interval
                processed = min(i*args.batch_size, args.N_train)
                writer.add_scalar('training_loss', cur_loss, processed)
                if args.print:
                    print(
                        (f"Train Epoch: {ep:2d}"
                        f"[{processed:6d}/{args.N_train:6d}"
                        f"({100.*processed/args.N_train:.0f}%)]"
                        f"\tLearning rate: {args.lr:.4f}\tLoss: {cur_loss:.6f}"))
                total_loss = 0
        """ EVALUATE """
        tcn.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0].to(device), data[1].to(device)
                output = tcn(x)
                test_loss = criterion(output, y)
                if args.print:
                    print(
                        f'\nTest set: Average loss: {test_loss.item():.6f}\n')
        writer.add_scalar('test_loss', test_loss.item() , ep)

    writer.close()
    torch.save(tcn.state_dict(), args.model_save_path)
    print('Finished Training')
    return 0

if __name__ == "__main__":
    run()

