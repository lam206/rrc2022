import config
from data import training_data, train_dataloader, test_data, test_dataloader


def evaluate(model, loss_fn):
    train_loss = 0
    scale = len(training_data) / config.BATCHSZ
    for obs, a in iter(train_dataloader):
        a_hat = model(obs.float().cuda())
        loss = loss_fn(a_hat, a.cuda())
        train_loss += loss / scale

    test_loss = 0
    scale = len(test_data) / config.BATCHSZ
    for obs, a in iter(test_dataloader):
        a_hat = model(obs.float().cuda())
        loss = loss_fn(a_hat, a.cuda())
        test_loss += loss / scale
    return train_loss, test_loss

