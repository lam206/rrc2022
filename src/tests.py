import gym

import rrc_2022_datasets

env = gym.make(
   # NOTE: There is a separate environment for each task and challenge stage.
   # See the documentation of the stages.
   "trifinger-cube-push-sim-expert-v0",
   disable_env_checker=True,
   visualization=True,  # enable visualization
)

dataset = env.get_dataset()

print("First observation: ", dataset["observations"][0])
print("First action: ", dataset["actions"][0])
print("First reward: ", dataset["rewards"][0])

