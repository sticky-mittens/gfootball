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
        'attack': {'pre': teleports.have_ball, 'post': lambda obs: teleports.not_have_ball(obs), 'OT': teleports.attack}, # 
        'defend': {'pre': teleports.not_have_ball, 'post': teleports.have_ball, 'OT': teleports.defend},
        },

    3 : {
        'charge_goal': {'pre': teleports.have_ball, 'post': teleports.almost_even_closer_to_goal, 'OT': teleports.charge_goal},
        'just_shoot': {'pre': teleports.have_ball, 'post': teleports.not_have_ball, 'OT': teleports.just_shoot},
        'maintain_ball_possession' : {'pre': teleports.have_ball, 'post': teleports.close_to_goal_in_x_axis, 'OT': teleports.maintain_ball_possession},
        'defend_': {'pre': teleports.not_have_ball, 'post': teleports.have_ball, 'OT': teleports.defend},
        },

    4 : {
        'charge_goal_': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': lambda obs: teleports.even_more_closer_to_goal(obs) or teleports.not_have_ball(obs), 'OT': teleports.charge_goal},
        'just_shoot_': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': teleports.not_have_ball, 'OT': teleports.just_shoot},
        'move_to_goal_with_wings': {'pre': teleports.have_ball, 'post': lambda obs: teleports.close_to_goal_in_x_axis(obs) or teleports.not_have_ball(obs), 'OT': teleports.move_to_goal_with_wings}, # lambda obs: teleports.close_to_goal_in_x_axis(obs) or teleports.not_have_ball(obs)
        'pass_to_a_player': {'pre': teleports.have_ball, 'post': lambda obs: teleports.have_ball(obs) or teleports.opponents_have_ball(obs), 'OT': teleports.pass_to_a_player},
        'defend__': {'pre': teleports.not_have_ball, 'post': teleports.have_ball, 'OT': teleports.defend},
        },

    5 : {
        'charge_goal__': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': lambda obs: teleports.even_more_closer_to_goal(obs) or teleports.not_have_ball(obs), 'OT': teleports.charge_goal},
        'just_shoot__': {'pre': lambda obs: teleports.have_ball(obs) and teleports.ball_in_opp_half(obs), 'post': teleports.not_have_ball, 'OT': teleports.just_shoot},
        'move_to_goal_with_wings_': {'pre': teleports.have_ball, 'post': lambda obs: teleports.close_to_goal_in_x_axis(obs) or teleports.not_have_ball(obs), 'OT': teleports.move_to_goal_with_wings},
        'pass_to_choice_1': {'pre': teleports.have_ball, 'post': teleports.have_ball, 'OT': lambda env, obs, obs_prev, done, ct: teleports.pass_to_choice(env, obs, obs_prev, done, ct, 1)},
        'pass_to_choice_2': {'pre': teleports.have_ball, 'post': teleports.have_ball, 'OT': lambda env, obs, obs_prev, done, ct: teleports.pass_to_choice(env, obs, obs_prev, done, ct, 2)},
        'pass_to_choice_3': {'pre': teleports.have_ball, 'post': teleports.have_ball, 'OT': lambda env, obs, obs_prev, done, ct: teleports.pass_to_choice(env, obs, obs_prev, done, ct, 3)},
        'pass_to_choice_4': {'pre': teleports.have_ball, 'post': teleports.have_ball, 'OT': lambda env, obs, obs_prev, done, ct: teleports.pass_to_choice(env, obs, obs_prev, done, ct, 4)},
        'pass_to_choice_5': {'pre': teleports.have_ball, 'post': teleports.have_ball, 'OT': lambda env, obs, obs_prev, done, ct: teleports.pass_to_choice(env, obs, obs_prev, done, ct, 5)},
        'defend___': {'pre': teleports.not_have_ball, 'post': teleports.have_ball, 'OT': teleports.defend},
        },
}

def deterministic_win_game(obs):
    if teleports.have_ball(obs):
        return 'attack'
    else:
        return 'defend'

def setup_D_initially():

    D = {
        'win_game' : {
                        'pre': all_OTs[1]['win_game']['pre'], 
                        'post': all_OTs[1]['win_game']['post'], 
                        'policy': deterministic_win_game, 
                        'deterministic': True
                    },
        'defend' : {
                        'pre': all_OTs[2]['defend']['pre'], 
                        'post': all_OTs[2]['defend']['post'], 
                        'policy': lambda obs: 'defend_', 
                        'deterministic': True
                    },
        'defend_' : {
                        'pre': all_OTs[2]['defend']['pre'], 
                        'post': all_OTs[2]['defend']['post'], 
                        'policy': lambda obs: 'defend__', 
                        'deterministic': True
                    },
        'charge_goal' : {
                        'pre': all_OTs[3]['charge_goal']['pre'], 
                        'post': all_OTs[3]['charge_goal']['post'], 
                        'policy': lambda obs: 'charge_goal_', 
                        'deterministic': True
                    },
        'just_shoot' : {
                        'pre': all_OTs[3]['just_shoot']['pre'], 
                        'post': all_OTs[3]['just_shoot']['post'], 
                        'policy': lambda obs: 'just_shoot_', 
                        'deterministic': True
                    },
        'move_to_goal_with_wings' : {
                        'pre': all_OTs[4]['move_to_goal_with_wings']['pre'], 
                        'post': all_OTs[4]['move_to_goal_with_wings']['post'], 
                        'policy': lambda obs: 'move_to_goal_with_wings_', 
                        'deterministic': True
                    }
    }

    return D


def convert_number_to_name(level_num, OT_num):
    if level_num == 1:
        if OT_num == 0:
            return 'defend'
        elif OT_num == 1:
            return 'attack'

    elif level_num == 2:
        if OT_num == 0:
            return 'charge_goal'
        elif OT_num == 1:
            return 'just_shoot'
        elif OT_num == 2:
            return 'maintain_ball_possession'
    
    elif level_num == 3:
        if OT_num == 0:
            return 'charge_goal'
        elif OT_num == 1:
            return 'just_shoot'
        elif OT_num == 2:
            return 'move_to_goal_with_wings'
        elif OT_num == 3:
            return 'pass_to_a_player'
    
    elif level_num == 4:
        return f'pass_to_choice_{OT_num + 1}'


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
        def shaped_reward_function(obs, obs_prev, chosen_OT_name, reward): # Alt: give (2-x dist from opp goal) as reward
            if chosen_OT_name == 'attack':
                if INBUILT:
                    return max(reward)
                else:
                    r = 0
                    if teleports.goal_scored(obs, obs_prev):
                        r += 1
                    elif teleports.opp_goal_scored(obs, obs_prev):
                        r += -1

                    if teleports.have_ball(obs) and teleports.ball_in_opp_half(obs):
                        r += 0.25
                    
                    #if teleports.have_ball(obs) and teleports.ball_in_opp_quarter(obs):
                    if teleports.have_ball(obs) and teleports.close_to_goal_in_x_axis(obs):
                        r += 0.25
                    
                    if teleports.have_ball(obs) and teleports.even_closer_to_goal(obs):
                        r += 0.5
                    return r
            else:
                if teleports.have_ball(obs):
                    return 1
                else:
                    return 0

    elif level_num == 2:
        def shaped_reward_function(obs, obs_prev, chosen_OT_name, reward):
            r = 0

            max_dist_to_opp_goal = teleports.dist(1, 0, -1, 0.42)
            # if max(reward) >= 0:
            #     print('env also gave high reward: ', reward)
            # if teleports.goal_scored(obs, obs_prev):
            #     print('We also believe goal was scored')
            #     r += 100        

            # if min(reward) < 0:
            #     print('env gave low reward: ', reward)
            #     if teleports.opp_goal_scored(obs, obs_prev):
            #         print('We also believe opp scored a goal.')
            #     r -= 1

            # if teleports.have_ball(obs): # and teleports.ball_in_opp_half(obs):
            #     # x_ball, y_ball, z_ball = obs[0]['ball']
            #     # r += 1 - teleports.dist(1, 0, x_ball, y_ball)**2/max_dist_to_opp_goal**2

            #     if teleports.close_to_goal_in_x_axis(obs) and chosen_OT_name == 'charge_goal':
            #         r += 0.1
                
            #     if teleports.almost_even_closer_to_goal(obs) and chosen_OT_name == 'just_shoot':
            #         r += 0.5 # more!

            # elif not teleports.have_ball(obs):
            #     # print('ball lost! penalizing...')
            #     r -= 0.25 # can remove

            return r

    elif level_num == 3:
        def shaped_reward_function(obs, obs_prev, chosen_OT_name, reward):
            r = 0
            if max(reward) >= 1:
                print('env gave high reward: ', reward)
                r += 3*max(reward)
            if min(reward) < 0:
                print('env gave low reward: ', reward)
                r -= 3*min(reward)

            if teleports.have_ball(obs) and teleports.ball_in_opp_half(obs):
                r += 0.25
            
            if teleports.have_ball(obs) and teleports.ball_in_opp_quarter(obs):
                r += 0.25
            
            if teleports.have_ball(obs) and teleports.even_closer_to_goal(obs):
                r += 0.5
            return r


    else: # level_num == 4
        def shaped_reward_function(obs, obs_prev, chosen_OT_name, reward):
            choice_num = float(chosen_OT_name.split('_')[-1])
            if teleports.opponents_have_ball(obs):
                return -1
            elif teleports.have_ball(obs):
                if teleports.choice_has_ball(obs, choice_num): # TODO convert choice_num to receiver_num and check if that person has ball
                    return 1
                return 0.5

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