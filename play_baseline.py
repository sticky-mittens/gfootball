import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import copy
import argparse
import os
from scripts import utils
from controllers import teleports

score_diffs = []
env = utils.get_env(render=False)
for ep_num in range(20):
    
    obs = env.reset()
    obs_prev = obs
    done = False
    ct = 0
    while not done:
        obs, obs_prev, reward, done, info, ct = teleports.common_sense_baseline(env, obs, obs_prev, done, ct)
        # print(f'env step {ct}')

    score_diff = obs[0]['score'][0] - obs[0]['score'][1]
    print(f'ep {ep_num} score_diff {score_diff}')
    score_diffs.append(score_diff)

avg_score_diff = sum(score_diffs)/len(score_diffs)
print(f'avg score diff after {len(score_diffs)} episodes is {avg_score_diff}')

