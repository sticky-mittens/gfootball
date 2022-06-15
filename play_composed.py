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
import gfootball.env as football_env
import random


def get_env(render, type, s, f_suffix):
    N_PLAYERS_LEFT = 11
    N_PLAYERS_RIGHT = 0

    if type == 'medium':
        env_str = '11_vs_11_stochastic'
    else:
        env_str = f'11_vs_11_{type}_stochastic'
    print(f'==> Playing in {env_str}.')

    env = football_env.create_environment(
        env_name=env_str, # 11_vs_11_easy_stochastic
        representation='raw',#'simple115v2',
        rewards='checkpoints,scoring',
        write_full_episode_dumps=True,
        write_video=True,
        logdir=f'./videos{f_suffix}/{type}_{s}/',
        number_of_left_players_agent_controls=N_PLAYERS_LEFT,
        number_of_right_players_agent_controls=N_PLAYERS_RIGHT,
        render=render,
        other_config_options={'physics_steps_per_frame': 3, 
                                'video_quality_level': 1,
                                'action_set': 'v2',
                                'display_game_stats': False
                            },
        )
    return env

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default = 2, help="2,3,5,10")
parser.add_argument("--mode", type=str, default = "easy", help="easy, medium, hard")
parser.add_argument("--v", type=int, default = 2, help="NA, 2, 3, 4, ... (videos folder suffix)")

args = parser.parse_args()

win_game_critic = torch.load(f'./full_run_logs/saved_models_lvl1_seed2_{args.mode}_0.9_1_0.001_20_2_3/win_game.pt')

if args.seed == 20:
    if args.v in [3,13]:
        suffix = {'easy': '_10_642', 'medium': '_7_1004', 'hard': '_4_100'}
    elif args.v in [2,12]:
        suffix = {'easy': '_9_606', 'medium': '_5_619', 'hard': '_3_629'}
    elif args.v in [1,11]:
        suffix = {'easy': '_10_209', 'medium': '_6_620', 'hard': '_4_826'}
else:
    suffix = {'easy': '', 'medium': '', 'hard': ''}
print(f'attack{suffix[args.mode]}.pt')

if args.mode == 'easy':
    print(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_easy_0.93_1.0_0.001_20.0_5_500/attack{suffix[args.mode]}.pt')
    attack_critic = torch.load(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_easy_0.93_1.0_0.001_20.0_5_500/attack{suffix[args.mode]}.pt') 
elif args.mode == 'medium':
    print(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_medium_0.93_10.0_0.01_10.0_6_500/attack{suffix[args.mode]}.pt')
    attack_critic = torch.load(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_medium_0.93_10.0_0.01_10.0_6_500/attack{suffix[args.mode]}.pt') 
elif args.mode == 'hard':
    print(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_hard_0.96_10.0_0.005_10.0_7_500/attack{suffix[args.mode]}.pt')
    attack_critic = torch.load(f'./full_run_logs/saved_models_lvl2_seed{args.seed}_hard_0.96_10.0_0.005_10.0_7_500/attack{suffix[args.mode]}.pt') 

torch.manual_seed(args.seed)
random.seed(args.seed)

learnt_options_set = utils.setup_D_initially()
learnt_options_set['win_game'] = {
    'pre': utils.all_OTs[1]['win_game']['pre'],
    'pre': utils.all_OTs[1]['win_game']['pre'],
    'policy' : win_game_critic,
    'deterministic': False
}
learnt_options_set['attack'] = {
    'pre': utils.all_OTs[2]['attack']['pre'],
    'pre': utils.all_OTs[2]['attack']['pre'],
    'policy' : attack_critic,
    'deterministic': False
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def select_action_epsilon_greedy(state, critic):
    sample = random.random()
    eps_threshold = 0.001
    if sample > eps_threshold:
        with torch.no_grad():
        # t.max(1) will return (largest column value of each row, its index).
        # we return this index of max element, reshaped/viewed as (1,1)
            return critic(state).max(1)[1].view(1, 1)
    else:
        return LongTensor([[random.randrange(3)]])

def select_action(state, critic):
    with torch.no_grad():
    # t.max(1) will return (largest column value of each row, its index).
    # we return this index of max element, reshaped/viewed as (1,1)
        return critic(state).max(1)[1].view(1, 1)

score_diffs = []

env = get_env(render=True, type=args.mode, s=args.seed, f_suffix=args.v)
for ep_num in range(25):
    obs = env.reset()
    obs_prev = obs
    done = False
    ct = 0
    while not done:
        obs_wrapped_lvl1 = utils.obs_wrapper(obs, 1)
        obs_wrapped_lvl1_tensor = FloatTensor(obs_wrapped_lvl1).to(device)

        lvl2_option_num = select_action(obs_wrapped_lvl1_tensor, win_game_critic)
        lvl2_option_num = lvl2_option_num.cpu().item()
        lvl2_option_name = utils.convert_number_to_name(1, lvl2_option_num)

        # lvl2_option_name = learnt_options_set['win_game']['policy'](obs)
        
        if lvl2_option_name == 'defend':
            obs, obs_prev, reward, done, info, ct = utils.all_OTs[3]['defend_']['OT'](env, obs, obs_prev, done, ct)
        elif lvl2_option_name == 'attack':            
            obs_wrapped_lvl2 = utils.obs_wrapper(obs, 2)
            obs_wrapped_lvl2_tensor = FloatTensor(obs_wrapped_lvl2).to(device)
            
            action_OT_number = select_action_epsilon_greedy(obs_wrapped_lvl2_tensor, attack_critic)
            action_OT_number = action_OT_number.cpu().item() 
            action_OT_name = utils.convert_number_to_name(2, action_OT_number)

            obs, obs_prev, reward, done, info, ct = utils.all_OTs[3][action_OT_name]['OT'](env, obs, obs_prev, done, ct)

    us = obs[0]['score'][0]
    opp = obs[0]['score'][1]
    score_diff = obs[0]['score'][0] - obs[0]['score'][1]
    print(f'ep {ep_num}, us {us}, opp {opp}, score_diff {score_diff}')
    score_diffs.append(score_diff)

avg_score_diff = sum(score_diffs)/len(score_diffs)
print(f'avg score diff after {len(score_diffs)} episodes is {avg_score_diff}')

