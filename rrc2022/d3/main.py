import d3rlpy
import gym
import rrc_2022_datasets

env = gym.make(
   "trifinger-cube-push-sim-expert-v0",
   disable_env_checker=True,
   visualization=False
)

dataset = env.get_dataset()

sac = d3rlpy.algos.SAC()

sac.fit(dataset, n_steps=10)




