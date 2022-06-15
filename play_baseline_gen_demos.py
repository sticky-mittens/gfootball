import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import time
import copy
import argparse
import os
from scripts import policies, utils
from controllers import teleports
import pickle

def save_to_transitions(obs_old_dict, action_int, reward_float, obs_dict):
    transitions.append((obs_old_dict, action_int, reward_float, obs_dict))



mem = policies.ReplayMemory(1000000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
def save_to_mem():
    print(f'Saving {len(transitions)} to memory')
    for tup in transitions:
        obs_old_dict, action_int, reward_float, obs_dict = tup
        mem.push((FloatTensor(utils.obs_wrapper(obs_old)).to(device),
                    LongTensor([[action_int]]).to(device), 
                    FloatTensor(utils.obs_wrapper(obs)).to(device),
                    FloatTensor([[reward_float]]).to(device) 
                    ))

shaped_reward_fn = utils.get_shaped_reward_fn(2)

for ep_num in range(200):
    env = utils.get_env(render=False)
    obs = env.reset()
    obs_prev = obs
    obs_old = obs.copy()
    done = False
    ct = 0
    transitions = []
    while not done:
        if not teleports.have_ball(obs):
            obs, obs_prev, reward, done, info, ct = teleports.defend(env, obs, obs_prev, done, ct)
        elif teleports.have_ball(obs):
            if teleports.close_to_goal_in_x_axis(obs):
                obs, obs_prev, reward, done, info, ct = teleports.charge_goal(env, obs, obs_prev, done, ct)
                save_to_transitions(obs_old, 0, shaped_reward_fn(obs, obs_prev, 'charge_goal', reward), obs)
                obs_old = obs.copy()
                
            if teleports.close_to_goal(obs, thresh=teleports.EVEN_CLOSER_TO_GOAL_THRESH):
                obs, obs_prev, reward, done, info, ct = teleports.just_shoot(env, obs, obs_prev, done, ct)
                save_to_transitions(obs_old, 1, shaped_reward_fn(obs, obs_prev, 'just_shoot', reward), obs)
                obs_old = obs.copy()

                continue

            obs, obs_prev, reward, done, info, ct = teleports.maintain_ball_possession(env, obs, obs_prev, done, ct)
            save_to_transitions(obs_old, 2, shaped_reward_fn(obs, obs_prev, 'maintain_ball_possession', reward), obs)
            obs_old = obs.copy()



    score_diff = obs[0]['score'][0] - obs[0]['score'][1]
    print(f'ep {ep_num} score_diff {score_diff}')
    if score_diff >= 3:
        save_to_mem()

save_loc = './demos.pkl'
with open(save_loc, 'wb') as file:
    pickle.dump(mem, file)
print(f'saved memory at {save_loc}')

