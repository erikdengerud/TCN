# run_electricity


if __name__ == "__main__":
    print("Importing modules.")
    from electricity.data import ElectricityDataSet
    from torch.utils.data import DataLoader

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    from tcn import TCN

    from electricity.metrics import WAPE, MAPE, SMAPE
    print("Modules imported.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    """ Dataset """
    train_start = '2012-01-01'
    train_end = '2014-05-30'
    test_start = '2014-06-30'
    test_end = '2014-12-30'

    v_batch_size = 10
    h_batch_size = 0

    print("Creating dataset.")
    train_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=train_start,
        end_date=train_end,
        h_batch=h_batch_size,
        include_time_covariates=True)
    test_dataset = ElectricityDataSet(
        'electricity/data/LD2011_2014_hourly.txt', 
        start_date=test_start,
        end_date=test_end,
        h_batch=0,
        include_time_covariates=True)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=v_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=v_batch_size, shuffle=True, num_workers=4)

    length_dataset = train_dataset.__len__()

    """ TCN """
    num_layers=5
    in_channels=8
    out_channels=1
    kernel_size=7
    residual_blocks_channel_size=[32, 32, 32, 32, 32]
    bias=True
    dropout=0.2
    stride=1
    leveledinit=True

    tcn = TCN(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            residual_blocks_channel_size=residual_blocks_channel_size,
            bias=bias,
            dropout=dropout,
            stride=1,
            dilations=None,
            leveledinit=False)

    tcn.to(device)

    pytorch_total_params = sum(
        p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")
    MODEL_SAVE_PATH = "models/tcn_elecctricity.pt"

    """ Training parameters"""
    epochs = 50
    lr = 0.0005
    momentum = 0.9
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tcn.parameters(), lr=lr)
    clip = False
    log_interval = 5

    """ Tensorboard """
    train_iter = iter(train_loader)
    x, y = train_iter.next()
    print(x.shape)
    print(y.shape)
    writer = SummaryWriter('runs/electricity_3')
    writer.add_graph(tcn, x)

    def train(epoch):
        global lr
        #tcn.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
        
            output = tcn(x)

            loss = criterion(output, y)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()

            if i % log_interval == 0 and i != 0:
                cur_loss = total_loss / log_interval
                processed = min(i*v_batch_size, length_dataset)
                writer.add_scalar(
                    'training_loss', cur_loss, processed+length_dataset*(epoch-1))
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, processed, length_dataset, 100.*processed/length_dataset, lr, cur_loss))
                total_loss = 0

    def evaluate():
        tcn.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0].to(device), data[1].to(device)
                output = tcn(x)
                test_loss = criterion(output, y)
                mape = MAPE(y, output)
                wape = WAPE(y, output)
                smape = SMAPE(y, output)
                print('\nTest set: Loss: {:.6f}\n'.format(test_loss.item()))
                print('\nTest set: WAPE: {:.6f}\n'.format(wape))
                print('\nTest set: MAPE: {:.6f}\n'.format(mape))
                print('\nTest set: SMAPE: {:.6f}\n'.format(smape))
                return test_loss.item(), wape, mape, smape

    print("Starting training.")
    for ep in range(1, epochs+1):
        train(ep)
        tloss, wape, mape, smape = evaluate()
        writer.add_scalar('test_loss', tloss , ep)
        writer.add_scalar('wape', wape , ep)
        writer.add_scalar('mape', mape , ep)
        writer.add_scalar('smape', smape , ep)

    writer.close()
    torch.save(tcn.state_dict(), MODEL_SAVE_PATH)
    print('Finished Training')





