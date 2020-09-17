import torch
import gym

import src.models as models

from src.env_utils import Environment, ActionActedWrapper, Minigrid2Image, VizdoomSparseWrapper, NoisyBackgroundWrapper, NoisyWallWrapper
import src.atari_wrappers as atari_wrappers
import vizdoomgym

import argparse
from os import path
import time

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')
parser.add_argument('--env', type=str, default='MiniGrid-ObstructedMaze-2Dlh-v0',
                    help='Gym environment. Other options are: SuperMarioBros-1-1-v0 \
                    or VizdoomMyWayHomeDense-v0 etc.')

parser.add_argument('--expe_path', type=str,
                    help='absolute path where model, optimizer etc.. are stored')

parser.add_argument('--noisy_wall', action='store_true')
parser.add_argument('--use_fullobs_policy', action='store_true')
parser.add_argument('--stop_visu', action='store_true')
parser.add_argument('--fix_seed', action='store_true')
parser.add_argument('--env_seed', default=1, type=int)



args = parser.parse_args()

action2name = dict([
    (0,'turn_left'),
    (1,'turn_right'),
    (2,'forward'),
    (3,'pickup'),
    (4, 'drop'),
    (5,'toggle'),
    (6, 'done')
])


is_minigrid = "MiniGrid" in args.env

if is_minigrid:
    env = gym.make(args.env)
    print(env)
    print(env.unwrapped.grid)
    env = Minigrid2Image(env)
    if args.noisy_wall:
        env = NoisyWallWrapper(env)
    env = ActionActedWrapper(env)
else:
    env = atari_wrappers.wrap_pytorch(
        atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, noop=False),
            clip_rewards=False,
            frame_stack=True,
            scale=False,
            fire=False))
    env = ActionActedWrapper(VizdoomSparseWrapper(env))

if 'MiniGrid' in args.env:
    if args.use_fullobs_policy:
        model = models.FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n)
    else:
        model = models.MinigridPolicyNet(env.observation_space.shape, env.action_space.n)

    embedder_model = models.MinigridStateEmbeddingNet(env.observation_space.shape)

else:
    model = models.MarioDoomPolicyNet(env.observation_space.shape, env.action_space.n)
    embedder_model = models.MarioDoomStateEmbeddingNet(env.observation_space.shape)

saved_checkpoint_path = path.join(args.expe_path, "model.tar")
checkpoint = torch.load(saved_checkpoint_path, map_location=torch.device('cpu'))

print(checkpoint['flags'])
if 'action_hist' in checkpoint:
    print(checkpoint["action_hist"])

model.load_state_dict(checkpoint['model_state_dict'])
model.train(False)

if 'state_embedding_model_state_dict' in checkpoint:
    embedder_model.load_state_dict(checkpoint['state_embedding_model_state_dict'])

env = Environment(env, fix_seed=args.fix_seed, env_seed=args.env_seed)
env_output = env.initial()
print(env.gym_env)


agent_state = model.initial_state(batch_size=1)
state_embedding = embedder_model(env_output['frame'])

if not args.stop_visu and is_minigrid:
    from gym_minigrid.window import Window
    w = Window(checkpoint['flags']['model'])
    arr = env.gym_env.render('rgb_array')
    #print("Arr", arr)
    w.show_img(arr)

while True :
    model_output, agent_state = model(env_output, agent_state)

    action = model_output["action"]
    #action = torch.randint(low=0, high=env.gym_env.action_space.n, size=(1,))
    env_output = env.step(action)

    next_state_embedding = embedder_model(env_output['frame'])

    #print(action2name[action.item()], torch.abs(state_embedding - next_state_embedding).sum())

    state_embedding = next_state_embedding

    if env_output['done']:
        agent_state = model.initial_state(batch_size=1)
        #print(env.env_seed)

    rgb_arr = env.gym_env.render('rgb_array')
    if not args.stop_visu and is_minigrid:
        w.show_img(rgb_arr)

    #print(env.gym_env)
    time.sleep(0.01)

