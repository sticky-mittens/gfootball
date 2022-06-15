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
import argparse
import time
import random

ep_map = {
    1: {'win_game': {'ep': 120}
        }, 
    2: {'attack': {'ep': 800}
        },
    3: {'maintain_ball_possession': {'ep': 4000}
        },
    4: {'pass_to_a_player': {'ep': 400}
        }
}

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default = 2, help="int seed")
parser.add_argument("--level", type=int, default = 1, help="level can be 1 or 2")
parser.add_argument("--deterministic_win_game", action='store_true', help="can choose determinstic to use determinstic win_game instead of loading net for faster runtime (since win_game's net is 100 percent accurate after training)")
parser.add_argument("--type", type=str, default = "easy", help="easy/medium/hard")
parser.add_argument("--gamma", type=float, default = 0.9, help="discount factor")
# parser.add_argument("--use_target", action='store_true', help="use target?")
# parser.add_argument("--target_update", type=float, default = 1, help="every how many episodes to do soft update?")
parser.add_argument("--learn_freq", type=float, default = 1, help="every how many episodes, should we do 10 learning steps (backprop on batch of transitions)?")
parser.add_argument("--lr", type=float, default = 0.001, help="critic learning rate")
parser.add_argument("--mem_size", type=float, default = 20, help="keep last how many best score_diff-ing episodes in replay buffer?")
parser.add_argument("--n_layers", type=int, default = 5, help="critic net layers")
parser.add_argument("--n_hidden_units", type=int, default = 500, help="critic net hidden units")

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

learnt_options_set = utils.setup_D_initially()
threshold_map = {'easy': 5, 'medium': 2, 'hard': 2}

current_env = utils.get_env_with_saved_videos(render=True, type=args.type, seed=args.seed) # ONLY New: thing
for level_num in [args.level]: # 1, 2
    print(f'==>Level {level_num}')
    print(f'==>OTs in this level are {utils.all_OTs[level_num+1].keys()}')
    shaped_reward_fn_for_this_level = utils.get_shaped_reward_fn(level_num)
    
    for OT_name in ep_map[level_num].keys():
        print(f'==>Currently learning {OT_name} to choose from above OTs.')
        if level_num == 1:
            current_policy = policies.Hierarchical_Policies(level_num, OT_name, folder_name = f'full_run_logs/saved_models_lvl{level_num}_seed{args.seed}_{args.type}_0.9_1_0.001_20_2_3', use_demos = False, gamma = 0.9, learn_freq = 1, critic_lr = 0.001, mem_size = 20, n_layers = 2, n_hidden_units = 3)
        elif level_num == 2:
            current_policy = policies.Hierarchical_Policies(level_num, OT_name, folder_name = f'full_run_logs/saved_models_with_videos_lvl{level_num}_seed{args.seed}_{args.type}_{args.gamma}_{args.learn_freq}_{args.lr}_{args.mem_size}_{args.n_layers}_{args.n_hidden_units}', use_demos = False, gamma = args.gamma, learn_freq = args.learn_freq, critic_lr = args.lr, mem_size = args.mem_size, n_layers = args.n_layers, n_hidden_units = args.n_hidden_units)
            
            if not args.deterministic_win_game:
                lvl1_seed = 2
                win_game_critic = torch.load(f'full_run_logs/saved_models_lvl1_seed{lvl1_seed}_{args.type}_0.9_1_0.001_20_2_3/win_game.pt', map_location=current_policy.device)
                learnt_options_set['win_game'] = {'pre': utils.all_OTs[1]['win_game']['pre'], 
                                            'post': utils.all_OTs[1]['win_game']['post'], 
                                            'policy': win_game_critic, 
                                            'deterministic': False
                                            }        
        
        # if not args.use_target:
        #     current_policy = policies.Hierarchical_Policies(level_num, OT_name, folder_name = f'saved_models_longRun_{args.type}_{args.gamma}_{args.learn_freq}', use_demos = False, gamma = args.gamma, learn_freq = args.leearn_freq)
        # else:
        #     current_policy = policies.Hierarchical_Policies(level_num, OT_name, folder_name = f'saved_models_longRun_{args.type}_{args.gamma}_{args.learn_freq}_useTarget_{args.target_update}', use_demos = False, gamma = args.gamma, learn_freq = args.leearn_freq, use_target = True, target_update = args.target_update)
        
        for ep_num in range(ep_map[level_num][OT_name]['ep']):
            sum_reward = current_policy.LearnGuidedPolicy(current_env, shaped_reward_fn_for_this_level, learnt_options_set, ep_num)
            
            current_policy.save_to_disk()
            
            # update learnt options set 
            learnt_options_set[OT_name] = {'pre': utils.all_OTs[level_num][OT_name]['pre'], 
                                            'post': utils.all_OTs[level_num][OT_name]['post'], 
                                            'policy': current_policy.critic, 
                                            'deterministic': False
                                            }
            
            if level_num == 2 and sum_reward >= threshold_map[args.type]:
                current_policy.save_to_disk_extra(sum_reward, ep_num)
            
            # if level_num == 1 and sum_reward >= current_policy.total_action_OT_calls_while_learning_for_lvl1-10:
            #     current_policy.save_to_disk_extra_lvl1(sum_reward, current_policy.total_action_OT_calls_while_learning_for_lvl1, ep_num)
        





