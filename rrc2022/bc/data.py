from torch.utils.data import Dataset, DataLoader
import gym
import rrc_2022_datasets
import config

class PushDataset(Dataset):
    def __init__(self, train):

        self.env = gym.make(
           "trifinger-cube-push-sim-expert-v0",
           disable_env_checker=True,
           visualization=True
        )
        tot_dataset = self.env.get_dataset()
        split = int(0.8*len(tot_dataset))
        if train:
            self.dataset = {key: tot_dataset[key][:split] for key in tot_dataset}
        else:
            self.dataset = {key: tot_dataset[key][split:] for key in tot_dataset}


    def __len__(self):
        return len(self.dataset["observations"])

    def __getitem__(self, idx):
        return self.dataset["observations"][idx], self.dataset["actions"][idx]


training_data = PushDataset(train=True)
train_dataloader = DataLoader(training_data, batch_size=config.BATCHSZ, shuffle=True)

test_data = PushDataset(train=False)
test_dataloader = DataLoader(test_data, batch_size=config.BATCHSZ, shuffle=True)



