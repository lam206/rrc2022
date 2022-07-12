from torch.nn import MSELoss
from model import BC
from torch.optim import SGD
from data import train_dataloader
import config
from evaluate import evaluate
import torch

model = BC()
loss_fn = MSELoss()
optim = SGD(model.parameters(), config.LR)

torch.save(model.state_dict(), "random.pt")

for e in range(config.EPOCHS):
    mse_train_loss = evaluate(model, loss_fn)
    print("MSE loss on training data: ", mse_train_loss)
    for obs, a in iter(train_dataloader):
        a_hat = model(obs.float())
        loss = loss_fn(a_hat, a)

        optim.zero_grad()
        loss.backward()
        optim.step()

torch.save(model.state_dict(), "overtrain.pt")

