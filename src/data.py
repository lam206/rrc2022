from torch.utils.data import Dataset, DataLoader
import gym
import rrc_2022_datasets

class PushDataset(Dataset):
    def __init__(self):

        self.env = gym.make(
           "trifinger-cube-push-sim-expert-v0",
           disable_env_checker=True,
           visualization=True
        )
        self.dataset = self.env.get_dataset()



    def __len__(self):
        return len(self.dataset["observations"])

    def __getitem__(self, idx):
        return self.dataset["observations"][idx], self.dataset["actions"][idx]


training_data = PushDataset()
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)


