import sys
sys.path.append("../")

from controllers import teleports
import gfootball.env as football_env
import numpy as np


all_OTs = {
    1 : {
        'win_game': {'pre': lambda x: True, 'post': lambda x: False, 'OT': None}
        },

    2 : {
        'charge_goal': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': teleports.almost_even_closer_to_goal, 'OT': teleports.charge_goal},
        'just_shoot': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': teleports.not_have_ball, 'OT': teleports.just_shoot},
        'maintain_ball_possession' : {'pre': teleports.have_ball, 'post': teleports.close_to_goal_in_x_axis, 'OT': teleports.maintain_ball_possession},
        'defend': {'pre': teleports.not_have_ball, 'post': teleports.have_ball, 'OT': teleports.defend},
        },
}

def deterministic_win_game(obs):
    if teleports.have_ball(obs):
        return 'attack'
    else:
        return 'defend'

def setup_D_initially():

    D = {}

    return D


def convert_number_to_name(level_num, OT_num):
    if level_num == 1:
        if OT_num == 0:
            return 'charge_goal'
        elif OT_num == 1:
            return 'just_shoot'
        elif OT_num == 2:
            return 'maintain_ball_possession'
        elif OT_num == 3:
            return 'defend'


def get_env(render=False, type='hard'):
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
        #write_full_episode_dumps=True,
        #write_video=True,
        #logdir='logs/',
        number_of_left_players_agent_controls=N_PLAYERS_LEFT,
        number_of_right_players_agent_controls=N_PLAYERS_RIGHT,
        render=render,
        other_config_options={#'physics_steps_per_frame': 3, 
                                #'video_quality_level': 1,
                                'action_set': 'v2',
                                #'display_game_stats': False
                            },
        )
    return env

def get_shaped_reward_fn(level_num):
    INBUILT = False
    if level_num == 1:
        def shaped_reward_function(obs, obs_prev, chosen_OT_name, reward): 
            return 0

    return shaped_reward_function

def do_flatten(obj): 
      """Run flatten on either python list or numpy array."""
      # from gfootball > env > wrappers.py
      if type(obj) == list:
        return np.array(obj).flatten()
      return obj.flatten()

def obs_wrapper(obs_input):
    '''
    takes in dict of all obs, returns list of essential obs
    '''
    # We oly need obs_input[0], nothing else is used in baseline.py or teleport.spy
    obs = obs_input[0]

    o = []
    o.extend(do_flatten(obs['left_team']))
    o.extend(do_flatten(obs['left_team_direction']))
    o.extend(do_flatten(obs['left_team_roles']))

    o.extend(do_flatten(obs['right_team']))
    o.extend(do_flatten(obs['right_team_direction']))
    o.extend(do_flatten(obs['right_team_roles']))


    # # If there were less than 11vs11 players we backfill missing values with
    # # -1.
    # # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
    # if len(o) < 88:   
    #     o.extend([-1] * (88 - len(o)))

    # ball position
    o.extend(obs['ball'])
    # ball direction
    o.extend(obs['ball_direction'])
    # one hot encoding of which team owns the ball
    if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
    if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
    if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])
    # one hot encoding of ball owned player : New
    owned = [0] * 11
    owned[obs['ball_owned_player']] = 1
    o.extend(owned)
    # score
    o.extend([obs['score'][0], obs['score'][1]])    
    

    # one hot encoding of actively controlled players
    # active = [0] * 11
    # if obs['active'] != -1:
    #     active[obs['active']] = 1
    # o.extend(active)

    game_mode = [0] * 7
    game_mode[obs['game_mode']] = 1
    o.extend(game_mode)

    return [o]

# def get_pre_post_from_action_OTs_of_previous_level(OT_name, level_num, full_map):
#     if level_num == 1:
#             pre_condition = lambda x: True
#             post_condition = lambda x: False
#     else:
#         options_of_prev_level = full_map[level_num-1]
#         for o in options_of_prev_level:
#             if OT_name in o['actions']:
#                 pre_condition, post_condition, _ = o['actions'][OT_name]
#     return pre_condition, post_condition


# full_ep_map = {
#     1: {'win_game': {'ep': 100, 'actions': get_action_OTs(1)['win_game']}
#         }, 
#     2: {'attack': {'ep': 400, 'actions': get_action_OTs(2)['attack']}, 
#         'defend': {'ep': 400, 'actions': get_action_OTs(2)['defend']}
#         }, 
#     3: {'pass_to_a_player': {'ep': 400, 'actions': get_action_OTs(3)['pass_to_a_player']}
#         }
# }


# def get_action_OTs(level_num):
#     if level_num == 1:
#         option_templates = {
#         'win_game': {
#             'attack': all_OTs['attack'],
#             'defend': all_OTs['defend']
#             }
#         }
#
#     elif level_num == 2:
#
#         option_templates = {
#         'attack': {
#             'charge_goal': all_OTs['charge_goal'],
#             'just_shoot': all_OTs['just_shoot'],
#             'move_to_goal_with_wings': all_OTs['move_to_goal_with_wings'],
#             'pass_to_a_player': all_OTs['pass_to_a_player']
#             },
#         }
#
#     else:
#         option_templates = {
#         'pass_to_a_player': {
#             'pass_to_choice_1': all_OTs['pass_to_choice_1'],
#             'pass_to_choice_2': all_OTs['pass_to_choice_2'],
#             'pass_to_choice_3': all_OTs['pass_to_choice_3'],
#             'pass_to_choice_4': all_OTs['pass_to_choice_4'],
#             'pass_to_choice_5': all_OTs['pass_to_choice_5']
#             }
#         }
#
#     return option_templates