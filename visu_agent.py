import torch
import gym

import src.models as models

from src.env_utils import Environment, ActionActedWrapper, Minigrid2Image

import argparse
from os import path
import time

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')
parser.add_argument('--env', type=str, default='MiniGrid-ObstructedMaze-2Dlh-v0',
                    help='Gym environment. Other options are: SuperMarioBros-1-1-v0 \
                    or VizdoomMyWayHomeDense-v0 etc.')

parser.add_argument('--expe_path', type=str,
                    help='absolute path where model, optimizer etc.. are stored')

parser.add_argument('--use_fullobs_policy', default=False)
parser.add_argument('--stop_visu', type=bool, default=False)

args = parser.parse_args()

env = ActionActedWrapper(Minigrid2Image(gym.make(args.env)))

if 'MiniGrid' in args.env:
    if args.use_fullobs_policy:
        model = models.FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n)
    else:
        model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n)

else:
    model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)

saved_checkpoint_path = path.join(args.expe_path, "model.tar")
checkpoint = torch.load(saved_checkpoint_path, map_location=torch.device('cpu'))

print(checkpoint['flags'])
if 'action_hist' in checkpoint:
    print(checkpoint["action_hist"])

model.load_state_dict(checkpoint['model_state_dict'])

env = Environment(env)
env_output = env.initial()
agent_state = model.initial_state(batch_size=1)

if not args.stop_visu:
    from gym_minigrid.window import Window
    w = Window(checkpoint['flags']['model'])
    arr = env.gym_env.render('rgb_array')
    w.show_img(arr)


while True :
    model_output, agent_state = model(env_output, agent_state)
    action = model_output["action"]
    env_output = env.step(action)

    if env_output['done']:
        agent_state = model.initial_state(batch_size=1)

    if not args.stop_visu:
        w.show_img(env.gym_env.render('rgb_array'))
        #time.sleep(1)

