# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gym
import torch 
from collections import deque, defaultdict
from gym import spaces
import numpy as np
from gym_minigrid.minigrid import OBJECT_TO_IDX, COLOR_TO_IDX

from gym_minigrid.envs import MultiRoomEnv
from gym_minigrid.register import register


def _format_observation(obs):
    obs = torch.tensor(obs)
    return obs.view((1, 1) + obs.shape) 


class VizdoomNormRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_reward=1000):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space
        self.last_obs = None
        self.max_reward = max_reward

    def reset(self):
        return self.env.reset()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)

        info["old_reward"] = reward
        reward = reward / self.max_reward

        return frame, reward, done, info


class VizdoomSparseWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space
        self.last_obs = None

    def reset(self):
        return self.env.reset()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)

        info["old_reward"] = reward

        if done:
            if self.unwrapped.game.is_player_dead() or self.unwrapped.game.get_episode_time() >= self.unwrapped.game.get_episode_timeout():
                #print("death or timeout  ", reward)
                reward = -1
            else:
                #assert reward > 900
                #print("victory  ", reward)
                reward = 1
        else:
            reward = 0

        return frame, reward, done, info


class ActionActedWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.observation_space = env.observation_space
        self.last_obs = None

    def reset(self):
        frame = self.env.reset()

        if frame.shape[0] == 4: # Vizdoom and frame stacking
            self.last_obs = frame[-1]
        else:
            self.last_obs = frame

        action_acted = True
        return frame, action_acted

    def step(self, action):
        frame, reward, done, info = self.env.step(action)

        if frame.shape[0] == 4: # Vizdoom and Frame stacking
            action_acted = not np.all(self.last_obs == frame[-1])
            self.last_obs = frame[-1]
        else:
            action_acted = not np.all(self.last_obs == frame)
            self.last_obs = frame

        return (frame, action_acted), reward, done, info

class NoisyBackgroundWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        rand_color = np.random.randint(len(COLOR_TO_IDX))
        observation[observation[:, :, 0] == 1, 1] = rand_color
        return observation

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
        return self.gym_env.env.env.gen_obs()['image']

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        self.episode_win = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        initial_action_acted = torch.ones(1, 1, dtype=torch.uint8)

        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)

        observation, acted = self.gym_env.reset()
        initial_frame = _format_observation(observation)
        partial_obs = _format_observation(self.get_partial_obs())

        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])   

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=self.episode_win,
            carried_col=carried_col,
            carried_obj=carried_obj,
            partial_obs=partial_obs,
            action_acted=initial_action_acted,
        )
        
    def step(self, action):
        (frame, action_acted), reward, done, _ = self.gym_env.step(action.item())

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
                self.gym_env.seed(seed=self.env_seed)
            frame, action_acted = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
            self.episode_win = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)
        action_acted = torch.tensor(action_acted).view(1, 1)

        partial_obs = _format_observation(self.get_partial_obs())
        
        if self.gym_env.env.env.carrying:
            carried_col, carried_obj = torch.LongTensor([[COLOR_TO_IDX[self.gym_env.env.env.carrying.color]]]), torch.LongTensor([[OBJECT_TO_IDX[self.gym_env.env.env.carrying.type]]])
        else:
            carried_col, carried_obj = torch.LongTensor([[5]]), torch.LongTensor([[1]])   

        return dict(
            frame=frame,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step = episode_step,
            episode_win = episode_win,
            carried_col = carried_col,
            carried_obj = carried_obj, 
            partial_obs=partial_obs,
            action_acted=action_acted,
            )

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


class OldEnvironment:
    def __init__(self, gym_env, fix_seed=False, env_seed=1):
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None
        self.fix_seed = fix_seed
        self.env_seed = env_seed

    def get_partial_obs(self):
        if hasattr(self.gym_env.unwrapped, "gen_obs"):
            return _format_observation(self.gym_env.env.env.gen_obs()['image'])
        else:
            return torch.zeros(1, 1)

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.ones(1, 1, dtype=torch.uint8)
        action_acted = torch.zeros(1, 1)

        if self.fix_seed:
            self.gym_env.seed(seed=self.env_seed)

        frame, acted = self.gym_env.reset()
        initial_frame = _format_observation(frame)

        return dict(
            frame=initial_frame,
            partial_obs=self.get_partial_obs(),
            reward=initial_reward,
            done=initial_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            action_acted=action_acted,
        )

    def step(self, action):
        (frame, acted), reward, done, _ = self.gym_env.step(action.item())

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += reward
        episode_return = self.episode_return

        if done:
            if self.fix_seed:
                self.gym_env.seed(seed=self.env_seed)
            frame, acted = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        frame = _format_observation(frame)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            frame=frame,
            partial_obs=self.get_partial_obs(),
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
            action_acted=acted,
        )

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


class MultiRoomEnvN7S4(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=7,
            maxNumRooms=7,
            maxRoomSize=4
        )


class MultiRoomEnvN10S10(MultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=10,
            maxNumRooms=10,
            maxRoomSize=10
        )

register(
    id='MiniGrid-MultiRoom-N7-S4-v0',
    entry_point='src.env_utils:MultiRoomEnvN7S4'
)

register(
    id='MiniGrid-MultiRoom-N10-S10-v0',
    entry_point='src.env_utils:MultiRoomEnvN10S10'
)