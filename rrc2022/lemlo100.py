import os, sys
import numpy as np
import torch
from rrc_2022_datasets import PolicyBase
from . import policies
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model import BC


class BCPolicy(PolicyBase):
    def __init__(self, action_space, observation_space, episode_length):
        # load torch script
        self.model = BC()
        self.model.cuda()
        torch_model_path = policies.get_model_path("overtrain.pt")
        print("Using ", torch_model_path)
        self.model.load_state_dict(torch.load(torch_model_path))    

    @staticmethod
    def is_using_flattened_observations():
        return True

    def reset(self):
        pass  # nothing to do here

    def get_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float)
        action = self.model(observation.unsqueeze(0).cuda()).cpu()
        print(action[0].detach())
        action = np.clip(action[0].detach(), -0.397, 0.397)
        return action.numpy()


