# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym
import torch 
from collections import deque, defaultdict

from gymnasium import spaces
import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

def _format_observation(obs):
    obs = torch.tensor(obs.copy()) # .copy() is to avoid "negative stride error" when converting numpy to torch tensor
    return obs.view((1, 1) + obs.shape) 

class Minigrid2Image(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, observation):
        return observation['image']


class Environment:
    def __init__(self, gym_env, fix_seed=False, env_seed=1):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.episode_win = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed

    def get_partial_obs(self):
        return self.gym_env.unwrapped.gen_obs()['image']

    def is_minigrid(self):
        """Check if environment is a MiniGrid environment."""
        return hasattr(self.gym_env.unwrapped, 'carrying')

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)

        if self.fix_seed:
            initial_state, info = self.gym_env.reset(seed=self.env_seed)
        else:
            initial_state, info = self.gym_env.reset()

        frame = _format_observation(initial_state["image"])

        # MiniGrid-specific features
        if self.is_minigrid():
            partial_obs = _format_observation(self.get_partial_obs())

            if self.gym_env.unwrapped.carrying:
                carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.unwrapped.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.unwrapped.carrying.type]]])
            else:
                carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])
        else:
            # Atari: use dummy values for MiniGrid-specific features
            partial_obs = torch.zeros(1, 7, 7, 3, dtype=torch.uint8)
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])

        buff = dict(
            frame=frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj,
            partial_obs=partial_obs
            )

        if "sound" in initial_state:
            buff["sound"] = _format_observation(initial_state["sound"])

        return buff
        
    def step(self, action):
        obs, reward, terminated, truncated, _ = self.gym_env.step(action.item())
        done = terminated or truncated

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        if done and reward > 0:
            self.episode_win[0][0] = 1
        else:
            self.episode_win[0][0] = 0
        episode_win = self.episode_win

        if done:
            if self.fix_seed:
                obs, _ = self.gym_env.reset(seed=self.env_seed)
            else:
                obs, _ = self.gym_env.reset()

            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_observation(obs["image"])
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        # MiniGrid-specific features
        if self.is_minigrid():
            partial_obs = _format_observation(self.get_partial_obs())

            if self.gym_env.unwrapped.carrying:
                carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.unwrapped.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.unwrapped.carrying.type]]])
            else:
                carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])
        else:
            # Atari: use dummy values for MiniGrid-specific features
            partial_obs = torch.zeros(1, 7, 7, 3, dtype=torch.uint8)
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])

        buff = dict(
                frame=frame,
                reward=reward,
                done=done,
                episode_return=episode_return,
                episode_step = episode_step,
                episode_win = episode_win,
                carried_col = carried_col,
                carried_obj = carried_obj,
                partial_obs=partial_obs
            )

        if "sound" in obs:
            sound = _format_observation(obs["sound"])
            buff["sound"] = sound

        return buff

    def get_full_obs(self):
        env = self.gym_env.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        return full_grid 
            
    def close(self):
        self.gym_env.close()


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


