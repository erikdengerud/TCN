# addtwo_run-py
"""
Train and test a TCN on the add two dataset.
Trying to reproduce https://arxiv.org/abs/1803.01271.
"""

class Network(object):
    def __init__(self):
        pass
    
    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

if __name__ == "__main__":
    from data import AddTwoDataSet
    from torch.utils.data import DataLoader

    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter

    from layers import TCN

    """ Dataset """
    N_train = 50000
    N_test = 1000
    seq_length = 200
    batch_size = 4

    train_dataset = AddTwoDataSet(N=N_train, seq_length=seq_length)
    test_dataset = AddTwoDataSet(N=N_test, seq_length=seq_length)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_iter = iter(train_loader)
    samples, labels = train_iter.next()

    """ TCN """
    num_layers=9
    in_channels=2
    out_channels=1
    kernel_size=3
    residual_blocks_channel_size=[16, 16, 16, 16, 16, 16, 16, 16, 1]
    bias=True
    dropout=0.2
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

    pytorch_total_params = sum(
        p.numel() for p in tcn.parameters() if p.requires_grad)
    print(f"Number of learnable parameters : {pytorch_total_params}")
    MODEL_SAVE_PATH = "models/tcn_addtwo_small.pt"

    """ Training parameters"""
    num_epochs = 300
    lr = 0.002
    momentum = 0.9
    criterion = nn.MSELoss()
    optimizer = optim.Adam(tcn.parameters(), lr=lr)

    """ Tensorboard """
    writer = SummaryWriter('runs/add_two_1')
    writer.add_graph(tcn, samples)


    """ Training """
    print("Starting training")
    running_loss = 0.0
    for epoch in range(1, num_epochs):

        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = tcn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # every 10 mini-batches...

                # ...log the running loss
                writer.add_scalar('training_loss',
                                running_loss / 100,
                                epoch * len(train_loader) + i)

                print(f"[{epoch:3} , {i+1:4}] : {running_loss}")
                running_loss = 0.0

    torch.save(tcn.state_dict(), MODEL_SAVE_PATH)
    writer.close()
    print('Finished Training')


    """ Model laod and test """
    tcn = TCN(*args, **kwargs)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = tcn(inputs)
            loss += criterion(outputs, inputs)

    print(f"Accuracy of the network on the test set: {loss}")




