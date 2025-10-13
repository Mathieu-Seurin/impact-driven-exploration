# Studying Sound Impact in Reinforcement Learning

## Installation

```
# create a new conda environment
conda create -n rlsound python=3.9
conda activate rlsound

# install impact-driven-rl
git clone git@github.com:Mathieu-Seurin/impact-driven-exploration.git
cd impact-driven-exploration
git checkout soundcur
pip install -r requirements.txt
cd ..

# install MiniGrid
git clone git@github.com:Mathieu-Seurin/minisound.git
cd minisound
pip install -e .

# install torch 1.13
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Checking if install works

```
OMP_NUM_THREADS=1 python main.py --model vanilla --env MiniGrid-MultiRoom-N2-S4-v0 --total_frames 1000000 --num_actors 3 --savedir outputs/
OMP_NUM_THREADS=1 python main.py --model ride --env MiniGrid-MultiRoom-N7-S4-v0 --total_frames 30000000 --intrinsic_reward_coef 0.1 --entropy_cost 0.0005 --num_actors 20
python visu_agent.py --expe_path $YOUR_EXPE --env MiniGrid-MultiRoom-N2-S4-v0

OMP_NUM_THREADS=1 python main.py --model vanilla --env ALE/Pong-v5 --total_frames 1000000 --num_actors 4 --savedir outputs/pong_sound```


## Acknowledgements
Our vanilla RL algorithm is based on [Torchbeast](https://github.com/facebookresearch/torchbeast), which is an open source implementation of IMPALA.

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.
