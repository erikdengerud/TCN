# addtwo_run-py
"""
Train and test a TCN on the add two dataset.
Trying to reproduce https://arxiv.org/abs/1803.01271.
"""


if __name__ == "__main__":
    from adding_problem.data import AddTwoDataSet
    from torch.utils.data import DataLoader

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    from tcn import TCN

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    """ Dataset """
    N_train = 50000
    N_test = 1000
    seq_length = 200
    batch_size = 32

    train_dataset = AddTwoDataSet(N=N_train, seq_length=seq_length)
    test_dataset = AddTwoDataSet(N=N_test, seq_length=seq_length)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    """ TCN """
    num_layers=8
    in_channels=2
    out_channels=1
    kernel_size=7
    residual_blocks_channel_size=[30, 30, 30, 30, 30, 30, 30, 30]
    bias=True
    dropout=0.0
    stride=1
    leveledinit=False

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
    MODEL_SAVE_PATH = "models/tcn_addtwo.pt"

    """ Training parameters"""
    epochs = 10
    lr = 0.002
    momentum = 0.9
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tcn.parameters(), lr=lr)
    clip = False
    log_interval = 100

    """ Tensorboard """
    train_iter = iter(train_loader)
    x, y = train_iter.next()
    writer = SummaryWriter('runs/add_two_1')
    writer.add_graph(tcn, x)

    def train(epoch):
        global lr
        tcn.train()
        total_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            output = tcn(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()

            if i % log_interval == 0:
                cur_loss = total_loss / log_interval
                processed = min(i*batch_size, N_train)
                writer.add_scalar('training_loss', cur_loss, processed)
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.4f}\tLoss: {:.6f}'.format(
                    epoch, processed, N_train, 100.*processed/N_train, lr, cur_loss))
                total_loss = 0

    def evaluate():
        tcn.eval()
        with torch.no_grad():
            for data in test_loader:
                x, y = data[0].cuda(), data[1].cuda()
                output = tcn(x)
                test_loss = F.mse_loss(output, y)
                print('\nTest set: Average loss: {:.6f}\n'.format(test_loss.item()))
                return test_loss.item()
    
    for ep in range(1, epochs+1):
        train(ep)
        tloss = evaluate()
        writer.add_scalar('test_loss', tloss , epoch)

    writer.close()
    torch.save(tcn.state_dict(), MODEL_SAVE_PATH)
    print('Finished Training')





