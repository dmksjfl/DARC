# Efficient Continuous Control with Double Actors and Regularized Critics

This repo contains our work in AAAI 2022, Efficient Continuous Control with Double Actors and Regularized Critics.

## Overview
`DDPG` is the fine-tuned version of vanilla DDPG which could achieve much better performance than `DDPG` at various environments. TD3 use fine-tuned `DDPG` as the baselines, and so we do in our work. The implementation of DARC is based on the open-source [TD3](https://github.com/sfujim/TD3) codebase.

We use `main.py` to run results where `DDPG.py` along with `TD3.py` are served as baselines and `DARC.py` is the core file for our work. We use seeds 1-5 for all algorithms during training and different seeds (the current seed + 100) during evaluation (see run.sh for more details).

## Evaluate True Value
One need to set the sampled state as the initial state in MuJoCo to evaluate the true value. Please refer to [https://github.com/openai/gym/issues/1617](https://github.com/openai/gym/issues/1617) for details.

## Requirements
- python: 3.7.9
- mujoco_py: 2.0.2.13
- torch: 1.8.0
- gym: 0.18.0
- box2d-py
- pybulletgym

## Install PybulletGym
Please refer to the open-source implementation of pybulletgym [here](https://github.com/benelot/pybullet-gym)

Before installing pybullet, make sure that you have gym installed. Then run the following commands to install  pybulletgym.
```
  git clone https://github.com/benelot/pybullet-gym.git
  cd pybullet-gym
  pip install -e .
```

### Use PubulletGym
```python
import pybulletgym
```
For detailed environments in pybulletgym, please refer [here](https://github.com/benelot/pybullet-gym).

## Usage
Utilize GPUs to accelerate training if available
```python
export CUDA_VISIBLE_DEVICES=1
```
Run the following commands to reproduce results in the submission
### Reproduce results in the submission
```python
./run.sh
```

### Run DARC
```python
python main.py --env <environment_name> --save-model --policy DARC --dir ./logs/DARC/r1 --seed 1 --qweight 0.1 --reg 0.005
```

### Run DDPG/TD3/DADDPG/DATD3
```python
python main.py --env <environment_name> --seed 1 --policy <algorithm_name> --dir './logs/' --save-model
```

## Citation
If you find our work helpful, please consider cite our work.
```
@inproceedings{Efficient2022Lyu,
  title={Efficient Continuous Control with Double Actors and Regularized Critics},
  author={Jiafei Lyu and Xiaoteng Ma and Jiangpeng Yan and Xiu Li},
  booktitle={Thirty-sixth AAAI Conference on Artificial Intelligence},
  year={2022},
}
```
