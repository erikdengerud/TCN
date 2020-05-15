# run_shapes

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import date, timedelta
import numpy as np
import pickle
import sys
from typing import List
import warnings

warnings.filterwarnings("ignore")

sys.path.append("")
sys.path.append("../../")

from data import ShapeDataset
from model import TCN
from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE
from utils.parser import parse, print_args
from utils.plot_predictions import plot_predictions


def train(epoch: int) -> None:
    tcn.train()
    epoch_train_loss = []
    total_loss = 0.0
    for i, d in enumerate(train_loader):
        x, y, idx, idx_row = (
            d[0].to(device),
            d[1].to(device),
            d[2].to(device),
            d[3].to(device),
        )

        optimizer.zero_grad()
        output = tcn(x, idx_row)
        # Since our x is longer than the y because we need the receptive field we
        # slice the output.
        output = output[:, :, -train_dataset.h_batch :]
        loss = criterion(output, y) / torch.abs(y).mean()
        loss.backward()
        if args.clip:
            for p in tcn.parameters():
                p.grad.data.clamp_(max=1e5, min=-1e5)
            # torch.nn.utils.clip_grad_norm_(tcn.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        epoch_train_loss.append(loss.item())

        if i % args.log_interval == 0 and i != 0:
            cur_loss = total_loss / (args.log_interval * args.v_batch_size)
            processed = min(i * args.v_batch_size, length_dataset)
            writer.add_scalar(
                "Loss/train", cur_loss, processed + length_dataset * epoch
            )
            if args.print:
                print(
                    (
                        f"Train Epoch: {epoch:2d}"
                        f"[{processed:6d}/{length_dataset:6d}"
                        f"({100.*processed/length_dataset:.0f}%)]"
                        f"\tLearning rate: {args.lr:.4f}\tLoss: {cur_loss:.6f}"
                    )
                )
            total_loss = 0
        if i == 0:
            total_loss = 0


def evaluate_final() -> List[float]:
    tcn.eval()
    with torch.no_grad():
        all_predictions = []
        all_real_values = []
        all_test_loss = []
        for i, data in enumerate(test_loader):
            x, y, idx, idx_row = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
            )

            predictions, real_values = tcn.rolling_prediction(
                x,
                idx_row,
                num_windows=args.num_rolling_periods,
                tau=args.length_rolling,
            )
            all_predictions.append(predictions)
            all_real_values.append(real_values)

            output = tcn(x, idx_row)
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
    torch.manual_seed(1729)
    np.random.seed(1729)

    args = parse()
    print_args(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    """ 
    Dataset 
    """
    print("Creating dataset.")
    """ Calculating start and end of test set based on the length of the receptive field """
    # Lookback of the TCN
    look_back = 1 + 2 * (args.kernel_size - 1) * 2 ** ((args.num_layers) - 1)
    print(f"Receptive field of the model is {look_back} time points.")
    # The timedelta function gives out days
    rolling_validation_length_days = timedelta(
        hours=args.num_rolling_periods * args.length_rolling
    )

    """ Training and test datasets """
    print("Train dataset")
    train_dataset = ShapeDataset(
        N=args.N,
        t=args.t,
        balance=False,
        predict_ahead=1,
        h_batch=args.h_batch_size,
        receptive_field=look_back,  # 3 layers, kernel size 7
        mean=args.mean,
        var=args.var,
    )
    df = train_dataset.df_descriptive.copy()
    print("Test dataset")
    test_dataset = ShapeDataset(
        N=args.N,
        t=args.t,
        balance=False,
        predict_ahead=1,
        h_batch=0,
        receptive_field=look_back,  # 3 layers, kernel size 7
        mean=args.mean,
        var=args.var,
        df=df,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.v_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.v_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    length_dataset = train_dataset.__len__()

    # Using the dimensions of the samples and labels as in and output channels in our model
    load_iter = iter(train_loader)
    x, y, _, _ = load_iter.next()
    in_channels = x.shape[1]
    out_channels = y.shape[1]
    """
    MODEL
    """
    """ TCN """
    tcn = TCN(
        num_layers=args.num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=args.kernel_size,
        residual_blocks_channel_size=[args.res_block_size] * args.num_layers,
        bias=args.bias,
        dropout=args.dropout,
        stride=args.stride,
        dilations=None,
        leveledinit=args.leveledinit,
        type_res_blocks=args.type_res_blocks,
        num_embeddings=train_dataset.N,
        embedding_dim=args.embedding_dim,
        embed=args.embed,
    )
    tcn.to(device)
    print(
        f"""Number of learnable parameters : {
            sum(p.numel() for p in tcn.parameters() if p.requires_grad)}"""
    )

    """ Training parameters"""
    criterion = nn.L1Loss()  # The criterion is scaled in the train function
    optimizer = optim.Adam(tcn.parameters(), lr=args.lr)

    """ Tensorboard """
    writer = SummaryWriter(log_dir=args.writer_path)

    """ 
    TRAINING
    """
    test_losses = []  # early stopping
    tenacity_count = 0
    for ep in range(1, args.epochs + 1):
        """ 
        To evaluate on only one random batch of the testset use evaluate(), 
        else use evaluate_final().
        """
        train(ep)
        with torch.no_grad():
            tloss, wape, mape, smape, mae, rmse = evaluate_final()
            writer.add_scalar("Loss/test", tloss, ep)
            writer.add_scalar("wape", wape, ep)
            writer.add_scalar("mape", mape, ep)
            writer.add_scalar("smape", smape, ep)
            writer.add_scalar("mae", mae, ep)
            writer.add_scalar("rmse", rmse, ep)
            fig = plot_predictions(
                tcn,
                test_loader,
                device,
                num_windows=args.num_rolling_periods,
                tau=args.length_rolling,
            )
            writer.add_figure("predictions", fig, global_step=ep)
            if args.print:
                print("Test set metrics:")
                print("Loss: {:.6f}".format(tloss.item()))
                print("WAPE: {:.6f}".format(wape))
                print("MAPE: {:.6f}".format(mape))
                print("SMAPE: {:.6f}".format(smape))
                print("MAE: {:.6f}".format(mae))
                print("RMSE: {:.6f}".format(rmse))

            # Visualizing embeddings
            if args.embed is not None:
                ids = [i for i in range(train_dataset.N)]
                ids = torch.LongTensor(ids).to(device)
                embds = tcn.embedding(ids)
                writer.add_embedding(
                    embds,
                    metadata=train_dataset.df_descriptive.to_numpy(),
                    global_step=ep,
                    tag="embedded id",
                )

        # Early stop
        if ep > args.tenacity + 1:
            if tloss < min(test_losses[-args.tenacity :]):
                tenacity_count = 0
            elif ep > args.tenacity:
                tenacity_count += 1
        test_losses.append(tloss)
        if tenacity_count >= args.tenacity:
            print("Early stop!")
            break

    tloss, wape, mape, smape, mae, rmse = evaluate_final()
    print("Test set:")
    print("Loss: {:.6f}".format(tloss))
    print("WAPE: {:.6f}".format(wape))
    print("MAPE: {:.6f}".format(mape))
    print("SMAPE: {:.6f}".format(smape))
    print("MAE: {:.6f}".format(mae))
    print("RMSE: {:.6f}".format(rmse))

    writer.close()
    # torch.save(tcn, args.model_save_path)
    torch.save(tcn.state_dict(), args.model_save_path)
    train_dataset.df_descriptive.to_csv("_".join([args.model_save_path, "df.csv"]))
    f = open("_".join([args.model_save_path, "_args.pkl"]), "wb")
    pickle.dump(args, f)
    f.close()
    print("Finished Training")
