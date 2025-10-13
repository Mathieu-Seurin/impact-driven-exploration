import torch

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

import src.models as models

from src.env_utils import Environment

import argparse
from os import path
import time
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')
parser.add_argument('--env', type=str, default='MiniGrid-',
                    help='Gym environment. Options: MiniGrid-*, ALE/Pong-v5, ALE/Breakout-v5, etc.')

parser.add_argument('--expe_path', type=str,
                    help='absolute path where model, optimizer etc.. are stored')

parser.add_argument('--noisy_wall', action='store_true')
parser.add_argument('--use_fullobs_policy', action='store_true')
parser.add_argument('--stop_visu', action='store_true')
parser.add_argument('--fix_seed', action='store_true')
parser.add_argument('--env_seed', default=1, type=int)

# Atari-specific arguments
parser.add_argument('--atari', action='store_true',
                    help='Use Atari environment (automatically detected from env name)')
parser.add_argument('--frame_stack', type=int, default=4,
                    help='Number of frames to stack for Atari (default: 4)')
parser.add_argument('--use_sound', action='store_true',
                    help='Use sound observations (for sound-enabled envs)')

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
is_atari = "ALE/" in args.env or args.atari or any(game in args.env for game in ["Pong", "Breakout", "SpaceInvaders", "Seaquest"])

if is_minigrid:
    env = gym.make(args.env, render_mode="human")
    if args.noisy_wall:
        from src.env_utils import NoisyWallWrapper
        env = NoisyWallWrapper(env)

elif is_atari:
    # Import Atari-specific wrappers from minisound
    import sys
    sys.path.insert(0, path.join(path.dirname(path.dirname(path.abspath(__file__))), 'minisound'))

    from minigrid.core.sound_processors import STFTSoundProcessor, NoSoundProcessor
    from minigrid.envs.atari_sound import make_atari_sound_env
    from minigrid.wrappers import AtariPreprocessingWrapper, FrameStackWrapper

    render_mode = "human" if not args.stop_visu else None

    # Create Atari environment with or without sound
    if args.use_sound:
        sound_processor = STFTSoundProcessor(sample_rate=16000)
    else:
        sound_processor = NoSoundProcessor()

    env = make_atari_sound_env(
        args.env,
        sound_processor=sound_processor,
        render_mode=render_mode
    )

    # Apply standard Atari preprocessing
    env = AtariPreprocessingWrapper(env, screen_size=84, grayscale=True, scale=True)
    env = FrameStackWrapper(env, num_stack=args.frame_stack)

else:
    raise ValueError(f"Unknown environment type: {args.env}")

if is_minigrid:
    if args.use_fullobs_policy:
        model = models.FullObsMinigridPolicyNet(env.observation_space.shape, env.action_space.n)
    else:
        if 'Sound' in args.env or 'sound' in args.env or args.use_sound:
            model = models.MinigridPolicyNet_Sound(env.observation_space, env.action_space.n)
        else:
            model = models.MinigridPolicyNet(env.observation_space, env.action_space.n)

    embedder_model = models.MinigridStateEmbeddingNet(env.observation_space)

elif is_atari:
    # Atari environments always use AtariPolicyNet_Sound architecture
    # (it handles both sound and no-sound cases via NoSoundProcessor)
    model = models.AtariPolicyNet_Sound(env.observation_space, env.action_space.n)

    # For embedder, use the image shape from observation space
    if hasattr(env.observation_space, 'spaces'):
        # Dict observation space (with sound)
        embedder_model = models.MarioDoomStateEmbeddingNet(env.observation_space["image"].shape)
    else:
        # Box observation space (no sound)
        embedder_model = models.MarioDoomStateEmbeddingNet(env.observation_space.shape)

else:
    # Default to MarioDoom models for other environments
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

print(env)
if is_minigrid:
    print(env.unwrapped.grid)

if hasattr(env, 'metadata'):
    env.metadata["render_fps"] = 30

env = Environment(env, fix_seed=args.fix_seed, env_seed=args.env_seed)
env_output = env.initial()
print(env.gym_env)


agent_state = model.initial_state(batch_size=1)
state_embedding = embedder_model(env_output['frame'])

# if not args.stop_visu and is_minigrid:
#     from minigrid.window import Window
#     w = Window(checkpoint['flags']['model'])
#     arr = env.gym_env.render('rgb_array')
#     #print("Arr", arr)
#     w.show_img(arr)

print(f"\nStarting visualization loop...")
print(f"Action space: {env.gym_env.action_space.n} actions")
print(f"Observation space: {env.gym_env.observation_space}\n")

step_count = 0
episode_count = 0
episode_reward = 0

while True:
    model_output, agent_state = model(env_output, agent_state)

    # action = model_output["action"]
    logits = model_output["policy_logits"]
    #print(logits)
    m = Categorical(logits=logits)
    action = m.sample()

    # action = torch.randint(low=0, high=env.gym_env.action_space.n, size=(1,))
    # action = torch.tensor([0])
    env_output = env.step(action)

    episode_reward += env_output['reward'].item()
    step_count += 1

    next_state_embedding = embedder_model(env_output['frame'])

    #print(action2name[action.item()], torch.abs(state_embedding - next_state_embedding).sum())

    state_embedding = next_state_embedding

    if env_output['done']:
        episode_count += 1
        print(f"Episode {episode_count} finished | Steps: {step_count} | Reward: {episode_reward:.2f}")
        agent_state = model.initial_state(batch_size=1)
        step_count = 0
        episode_reward = 0
        #print(env.env_seed)

    # Render environment
    if not args.stop_visu:
        if is_minigrid:
            env.gym_env.render()
        elif is_atari:
            # Atari rendering is handled automatically with render_mode="human"
            pass

    #print(env.gym_env)
    #time.sleep(0.001)
