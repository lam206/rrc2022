from torch.nn import MSELoss
from model import BC
from torch.optim import Adam
from data import train_dataloader, test_dataloader, training_data, test_data
import config
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import shutil


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ExperimentTracker():

    def __init__(self):
        now = datetime.now()
        current_time = now.strftime("%y-%m-%d-%H-%M-%S")
        self.dir = "experiments/experiment" + current_time

        self.tb_writer = SummaryWriter(self.dir)

        shutil.copy("config.py", self.dir)

    def log_evaluation(self, model, loss_fn, step):
        mse_train_loss, mse_test_loss = self.evaluate(model, loss_fn)
        self.tb_writer.add_scalar("mse_train_loss", mse_train_loss, step)
        self.tb_writer.add_scalar("mse_test_loss", mse_test_loss, step)

    def evaluate(self, model, loss_fn):
        train_loss = 0
        scale = len(training_data) / config.BATCHSZ
        for obs, a in iter(train_dataloader):
            a_hat = model(obs.float().to(device))
            loss = loss_fn(a_hat, a.to(device))
            train_loss += loss / scale

        test_loss = 0
        scale = len(test_data) / config.BATCHSZ
        for obs, a in iter(test_dataloader):
            a_hat = model(obs.float().to(device))
            loss = loss_fn(a_hat, a.to(device))
            test_loss += loss / scale
        return train_loss, test_loss



if __name__ == '__main__':
    model = BC()
    model.to(device)
    loss_fn = MSELoss()
    optim = config.OPTIMISER(model.parameters(), lr=config.LR)

    tracker = ExperimentTracker()
    
    for e in range(config.EPOCHS):
        print("Epoch ", e)
        tracker.log_evaluation(model, loss_fn, e)
        
        for obs, a in iter(train_dataloader):
            a_hat = model(obs.float().to(device))
            loss = loss_fn(a_hat, a.to(device))

            optim.zero_grad()
            loss.backward()
            optim.step()


