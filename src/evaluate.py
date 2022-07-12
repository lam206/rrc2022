import config
from data import training_data, train_dataloader


def evaluate(model, loss_fn):
    mse_loss = 0
    scale = len(training_data) / config.BATCHSZ
    for obs, a in iter(train_dataloader):
        a_hat = model(obs.float())
        loss = loss_fn(a_hat, a)
        mse_loss += loss / scale
    return mse_loss

