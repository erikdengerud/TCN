# run_electricity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datetime import date, timedelta
import numpy as np
import pandas as pd
import sys
from typing import List
import warnings
import pickle

warnings.filterwarnings("ignore")

sys.path.append("")
sys.path.append("../../")

from data import RevenueDataset
from model import TCN
from utils.metrics import WAPE, MAPE, SMAPE, MAE, RMSE
from utils.parser import parse, print_args
from utils.plot_predictions import plot_predictions


def train(epoch: int) -> None:
    tcn.train()
    epoch_train_loss = []
    total_loss = 0.0
    for i, d in enumerate(train_loader):
        x, y, idx, idx_row, id_sect = (
            d[0].to(device),
            d[1].to(device),
            d[2].to(device),
            d[3].to(device),
            d[4].to(device),
        )

        optimizer.zero_grad()
        emb = idx_row if not args.embed_sector else id_sect
        output = tcn(x, emb)
        # Since our x is longer than the y because we need the receptive field we
        # slice the output.
        output = output[:, :, -train_dataset.h_batch :]
        loss = criterion(output, y) / torch.abs(y).mean()
        loss.backward()
        if args.clip:
            for p in tcn.parameters():
                p.grad.data.clamp_(max=1e5, min=-1e5)
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
            x, y, idx, idx_row, id_sect = (
                data[0].to(device),
                data[1].to(device),
                data[2].to(device),
                data[3].to(device),
                data[4].to(device),
            )

            if args.data_scale:
                x_np = np.zeros((test_dataset.num_ts, test_dataset.length_ts))
                idx_row_list = idx_row.detach().cpu().numpy()
                if (
                    args.cluster_covariate
                    or args.zero_covariate
                    or args.random_covariate
                ):
                    x_np[idx_row_list] = x[:, 0, :].squeeze().cpu().numpy()
                    x_scaled = test_dataset.data_scaler.transform(x_np.T).T
                    x_scaled = (
                        torch.from_numpy(x_scaled).to(dtype=torch.float32).unsqueeze(1)
                    )
                    x_scaled = x_scaled[idx_row_list].to(device)
                    x_scaled = torch.cat((x_scaled, x[:, 1, :].unsqueeze(1)), 1).to(
                        device
                    )

                else:
                    x_np[idx_row_list] = x.squeeze().cpu().numpy()
                    x_scaled = test_dataset.data_scaler.transform(x_np.T).T
                    x_scaled = (
                        torch.from_numpy(x_scaled).to(dtype=torch.float32).unsqueeze(1)
                    )
                    x_scaled = x_scaled[idx_row_list].to(device)

                predictions, _ = tcn.rolling_prediction(
                    x_scaled,
                    emb_id=idx_row if not args.embed_sector else id_sect,
                    tau=args.length_rolling,
                    num_windows=args.num_rolling_periods,
                )

                predictions_rescaled = np.zeros(
                    (
                        test_dataset.num_ts,
                        args.num_rolling_periods * args.length_rolling,
                    )
                )
                predictions_rescaled[idx_row_list] = predictions.cpu().numpy()
                predictions_rescaled = test_dataset.data_scaler.inverse_transform(
                    predictions_rescaled.T
                ).T
                predictions = predictions_rescaled[idx_row_list]
                predictions = torch.from_numpy(predictions).to(dtype=torch.float32)
                real_values = x[:, 0, -args.num_rolling_periods * args.length_rolling :]

                all_predictions.append(predictions)
                all_real_values.append(real_values)

                emb = idx_row if not args.embed_sector else id_sect

                output = tcn(x_scaled, emb)
                output = output[:, :, -args.num_rolling_periods * args.length_rolling :]
                output_rescaled = np.zeros(
                    (
                        test_dataset.num_ts,
                        args.num_rolling_periods * args.length_rolling,
                    )
                )
                output_rescaled[idx_row_list] = output.squeeze().cpu().numpy()
                output_rescaled = test_dataset.data_scaler.inverse_transform(
                    output_rescaled.T
                ).T
                output = output_rescaled[idx_row_list]
                output = torch.from_numpy(output).to(dtype=torch.float32, device=device)

                y = y[:, :, -args.num_rolling_periods * args.length_rolling :]

                test_loss = criterion(output, y) / torch.abs(y).mean()
                all_test_loss.append(test_loss.item())
            else:
                predictions, real_values = tcn.rolling_prediction(
                    x,
                    emb_id=idx_row if not args.embed_sector else id_sect,
                    tau=args.length_rolling,
                    num_windows=args.num_rolling_periods,
                )
                all_predictions.append(predictions)
                all_real_values.append(real_values)

                emb = idx_row if not args.embed_sector else id_sect
                output = tcn(x, emb)
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
    if args.dilated_convolutions:
        look_back = 1 + 2 * (args.kernel_size - 1) * 2 ** (args.num_layers - 1)
    else:
        look_back = 1 + (args.kernel_size - 1) * args.num_layers
    print(f"Receptive field of the model is {look_back} time points.")
    look_back_timedelta = timedelta(hours=look_back)

    # Quarter train end
    test_start = (
        pd.to_datetime(args.train_end) - look_back * pd.offsets.QuarterEnd()
    ).strftime("%Y-%m-%d")
    test_end = (
        pd.to_datetime(args.train_end)
        + args.num_rolling_periods * args.length_rolling * pd.offsets.QuarterEnd()
    ).strftime("%Y-%m-%d")

    print(args.train_end)
    print(args.train_start)
    print(test_start)
    print(test_end)

    """ Training and test datasets """
    print("Train dataset")
    train_dataset = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        data_scale=args.data_scale,
        data_scaler=None,
        start_date=args.train_start,
        end_date=args.train_end,
        h_batch=args.h_batch_size,
        receptive_field=look_back,
        cluster_covariate=args.cluster_covariate,
        random_covariate=args.random_covariate,
        zero_covariate=args.zero_covariate,
        cluster_dict_path=args.cluster_dict_path,
    )
    print("Test dataset")
    test_dataset = RevenueDataset(
        file_path="revenue/data/processed_companies.csv",
        meta_path="revenue/data/comp_sect_meta.csv",
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        receptive_field=look_back,
        data_scale=False,
        data_scaler=train_dataset.data_scaler,
        cluster_covariate=args.cluster_covariate,
        random_covariate=args.random_covariate,
        zero_covariate=args.zero_covariate,
        cluster_dict_path=args.cluster_dict_path,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.v_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    # If we use clustering covariates we have to recalculate the prototypes at each time
    # step. We therefore need the full dataset as batch size. This should be done in
    # another way if the dataset is huge.
    v_test_batch = test_dataset.num_ts if args.cluster_covariate else args.v_batch_size
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=v_test_batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    length_dataset = train_dataset.__len__()
    length_train = train_dataset.length_ts
    length_test = test_dataset.length_ts
    print(length_train)
    print(length_test)

    # Using the dimensions of the samples and labels as in and output channels in our model
    load_iter = iter(train_loader)
    x, y, _, _, _ = load_iter.next()
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
        num_embeddings=train_dataset.num_ts
        if not args.embed_sector
        else train_dataset.num_sect,
        embedding_dim=args.embedding_dim,
        embed=args.embed,
        dilated_convolutions=args.dilated_convolutions,
        clustering_dict=train_dataset.cluster_dict if args.cluster_covariate else None,
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
        print("Train")
        train(ep)
        print("Evaluate")
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
                embed_sect=args.embed_sector,
                num_windows=args.num_rolling_periods,
                tau=args.length_rolling,
            )
            writer.add_figure("predictions", fig, global_step=ep)

            """
            # Visualizing embeddings
            if args.embed is not None:
                if args.embed_sector:
                    ids = [i for i in range(train_dataset.num_sect)]
                    ids = torch.LongTensor(ids).to(device)
                    d = {v: k for k, v in train_dataset.sect_id_dict.items()}
                    meta_sector = [d[id.item()] for id in ids]
                    embds = tcn.embedding(ids)
                    writer.add_embedding(
                        embds, metadata=meta_sector, global_step=ep, tag="embedded id"
                    )
                else:
                    ids = [i for i in range(train_dataset.num_ts)]
                    ids = torch.LongTensor(ids).to(device)
                    meta_sector = [
                        train_dataset.comp_sect_dict[
                            #train_dataset.companies_id_dict[id.item()]
                            train_dataset.id_companies_dict[id.item()]
                        ][0]
                        for id in ids
                    ]
                    embds = tcn.embedding(ids)
                    writer.add_embedding(
                        embds, metadata=meta_sector, global_step=ep, tag="embedded id"
                    )
            """

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

    print("Evaluate final")
    tloss, wape, mape, smape, mae, rmse = evaluate_final()
    print("Test set:")
    print("Loss: {:.6f}".format(tloss))
    print("WAPE: {:.6f}".format(wape))
    print("MAPE: {:.6f}".format(mape))
    print("SMAPE: {:.6f}".format(smape))
    print("MAE: {:.6f}".format(mae))
    print("RMSE: {:.6f}".format(rmse))

    writer.close()

    if args.embed is not None:
        if args.embed_sector:
            ids = [i for i in range(train_dataset.num_sect)]
            ids = torch.LongTensor(ids).to(device)
            d = {v: k for k, v in train_dataset.sect_id_dict.items()}
            meta_sector = [d[id.item()] for id in ids]
            embds = tcn.embedding(ids).detach().cpu().numpy()
            np.save(
                f"representations/representation_matrices/revenue{'_scaled_' if args.data_scale else '_'}embedded_sector_nc_{args.embedding_dim}.npy",
                embds,
            )
        else:
            ids = [i for i in range(train_dataset.num_ts)]
            ids = torch.LongTensor(ids).to(device)
            meta_sector = [
                train_dataset.comp_sect_dict[
                    # train_dataset.companies_id_dict[id.item()]
                    train_dataset.id_companies_dict[id.item()]
                ][0]
                for id in ids
            ]
            embds = tcn.embedding(ids).detach().cpu().numpy()
            np.save(
                f"representations/representation_matrices/revenue{'_scaled_' if args.data_scale else '_'}embedded_id_nc_{args.embedding_dim}.npy",
                embds,
            )

    # torch.save(tcn, args.model_save_path)
    torch.save(tcn.state_dict(), args.model_save_path)
    f = open("_".join([args.model_save_path, "_args.pkl"]), "wb")
    pickle.dump(args, f)
    f.close()
    print("Finished Training")

# python revenue/run_revenue.py --num_workers 0 --model_save_path revenue/models/test_local --writer_path revenue/runs/test_local --epochs 1 --tenacity 20 --clip --log_interval 100 --print --train_start 2007-01-01 --train_end 2017-01-01 --num_rolling_periods 2 --length_rolling 4 --v_batch_size 32 --h_batch_size 3 --num_layers 2 --kernel_size 2 --res_block_size 4 --embed post --embedding_dim 2
