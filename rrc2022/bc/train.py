from torch.nn import MSELoss
from model import BC
from torch.optim import Adam
from data import train_dataloader
import config
from evaluate import evaluate
import torch

model = BC()
model.cuda()
loss_fn = MSELoss()
optim = Adam(model.parameters(), config.LR)

for e in range(config.EPOCHS):
    mse_train_loss, mse_test_loss = evaluate(model, loss_fn)
    print("MSE loss on training data: ", mse_train_loss)
    print("MSE loss on test data: ", mse_test_loss)
    torch.save(model.state_dict(), "adam_epochs:" + str(e) + ",test_loss:" + str(mse_test_loss) + ".pt")

    for obs, a in iter(train_dataloader):
        a_hat = model(obs.float().cuda())
        loss = loss_fn(a_hat, a.cuda())

        optim.zero_grad()
        loss.backward()
        optim.step()


