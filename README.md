# Discovering Diverse Solutions in Deep Reinforcement Learning by Maximizing State-Action-Based Mutual Information

This repository contains authors' implementation of LTD3.
LTD3 is an algorithm to train a latent-conditioned policy that models diverse solutions. 
[links: [paper](https://www.sciencedirect.com/science/article/pii/S0893608022001393?via%3Dihub), [arXiv](https://arxiv.org/abs/2103.07084)]


## Requirements
gym == 0.9.1 \
mujoco_py == 0.5.7 \
numpy == 1.19.4 \
torch == 1.9.0 \
torchaudio == 0.9.0 \
torchvision == 0.10.0  

## How to run the code
To use the cusomized environment, it is necessary to copy the xml files such as "walker2d_lowshort.xml" in "velEnv" and paste them in the folder "site-packages\gym\envs\mujoco\assets" in your anaconda environment.

To see the behaviors of a policy trained on the Walker2DVel, run
```
python visualize_policy_LTD3_cont2d.py 
```
To train a policy with LTD3, run
```
python LTD3_training.py 
```
To perform few-shot adaptation using a pre-trained policy, run
```
python few_shot_adaptation_LTD3.py 
```
