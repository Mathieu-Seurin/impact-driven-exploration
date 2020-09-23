#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

from gym_minigrid.minigrid import *

from os import path
import io

import cv2

import torch

import vizdoomgym
from src.env_utils import ActionActedWrapper, Minigrid2Image, NoisyBackgroundWrapper, Environment, PlayGround

import matplotlib.pyplot as plt
import seaborn as sns

from src import models


class TwoCorridor(MiniGridEnv):
    def __init__(self,
                 size=8,
                 agent_start_pos=(1, 1),
                 agent_start_dir=0,
                 ):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)

        self.put_obj(Ball('green'), 2, 1)


        for i in range(6):
            self.put_obj(Wall(), 3, i + 1)

        self.put_obj(Door('blue'), 3, 5)

        self.put_obj(Ball('red'), 4, 1)
        self.put_obj(Key('green'), 4, 2)
        self.put_obj(Box('grey'), 4, 3)
        self.put_obj(Ball('blue'), 4, 4)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"



def get_img_from_fig(fig, dpi=180):

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def redraw(img):
    if not args.agent_view:
        img = env.gym_env.render('rgb_array')#, tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs, acted = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    obs = obs.astype(np.float)
    redraw(obs)

def step(action):
    global current_state, next_state

    env_obs = env.step(torch.tensor((action,)))
    # print('step=%s, reward=%.2f' % (env.step_count, reward))

    next_state = env_obs["frame"]

    x, y = env.gym_env.agent_pos

    if env_obs['action_acted']:
        action_rew = 1 - (hists['acted'][action] / hists['usage'][action])
    else:
        action_rew = 0

    ride_rew = torch.abs(embedder_model(current_state) - embedder_model(next_state)).sum()

    hmap_action[y, x] += action_rew
    hmap_ride[y, x] += ride_rew
    count_state[y, x] += 1

    if env_obs['done'] or action == env.gym_env.actions.done :
        print('done!')

        #final_heat_action = hmap_action / count_state
        #final_heat_ride = hmap_ride / count_state
        final_heat_action = hmap_action
        final_heat_ride = hmap_ride

        fig = plt.figure()
        ax = sns.heatmap(final_heat_action)
        heatmap_action_pixel = get_img_from_fig(fig)

        fig = plt.figure()
        ax = sns.heatmap(final_heat_ride)
        heatmap_ride_pixel = get_img_from_fig(fig)

        np.save('background.npy', np.asarray(background))
        np.save('heatmap_action.npy', heatmap_action_pixel)
        np.save('heatmap_ride.npy', heatmap_ride_pixel)

        quit()
        #reset()
    else:
        redraw(env_obs['partial_obs'])

def key_handler(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    # if event.key == 'backspace':
    #     reset()
    #     return
    if event.key == 'left':
        step(env.gym_env.actions.left)
        return
    if event.key == 'right':
        step(env.gym_env.actions.right)
        return
    if event.key == 'up':
        step(env.gym_env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.gym_env.actions.toggle)
        return
    if event.key == 'enter':
        step(env.gym_env.actions.pickup)
        return
    if event.key == 'delete':
        step(env.gym_env.actions.drop)
        return

    if event.key == 'f':
        step(env.gym_env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument("--env", help="gym environment to load", default='MiniGrid-MultiRoom-N6-v0')
parser.add_argument("--seed", type=int, help="random seed to generate the environment with", default=-1)
parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32)
parser.add_argument('--agent_view', default=False, help="draw the agent sees (partially observable view)", action='store_true')

parser.add_argument('--hist_path', default="results/dummy/torchbeast-20200908-162539")
parser.add_argument('--ride_path', default="results/ObstructedMaze-2Dlh/torchbeast-20200813-103046")

args = parser.parse_args()


if "MiniGrid" in args.env:
    # env = ActionActedWrapper(Minigrid2Image(TwoCorridor(8)))
    env = ActionActedWrapper(Minigrid2Image(PlayGround(16, agent_start_pos=(8,8))))
    env = Environment(env)
else:
    raise NotImplementedError("Minigrid only is available")

if args.hist_path :
    hist_path = path.join(args.hist_path, 'action_hist.tar')
    if path.exists(hist_path):
        hists = torch.load(hist_path)
    else:
        quit()  # todo add all possible model here

if args.ride_path:
    embedder_model = models.MinigridStateEmbeddingNet(env.gym_env.observation_space.shape)
    saved_checkpoint_path = path.join(args.ride_path, "model.tar")
    checkpoint = torch.load(saved_checkpoint_path, map_location=torch.device('cpu'))
    embedder_model.load_state_dict(checkpoint['state_embedding_model_state_dict'])

hists = hists[0]
initial_state = env.initial()

current_state = initial_state['frame']

print(env.gym_env.width, env.gym_env.width)

#global hmap, count_state
count_state = np.ones((env.gym_env.width, env.gym_env.width))
hmap_action = np.zeros((env.gym_env.width, env.gym_env.width))
hmap_ride = np.zeros((env.gym_env.width, env.gym_env.width))

background = None

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

background = env.gym_env.render('rbg_array')

redraw(background)

# Blocking event loop
window.show(block=True)
