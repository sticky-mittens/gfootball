import gfootball.env as football_env
import random
import numpy as np
import torch
import math

DIST_TO_BALL_THRESH = 0.1
DIST_TO_OPP_THRESH = 0.3
SHORT_LONG_THRESH = 0.4
HIGH_THRESH = 1
N_PLAYERS_LEFT = 11 # change according to scenario below
N_PLAYERS_RIGHT = 0 # num of opponent team that we control

env = football_env.create_environment(
    env_name='11_vs_11_easy_stochastic',#'academy_3_vs_1_with_keeper',#'11_vs_11_easy_stochastic',#
    representation='raw',#'simple115v2',
    rewards='checkpoints,scoring',
    write_full_episode_dumps=True,
    write_video=True,
    logdir='logs/',
    number_of_left_players_agent_controls=N_PLAYERS_LEFT,
    number_of_right_players_agent_controls=N_PLAYERS_RIGHT,
    render=False,
    other_config_options={'physics_steps_per_frame': 4, 
                            'video_quality_level': 1,
                            'display_game_stats': False
                        },
    )

action_map = {'left':           {'action': 1, 'signs': (-1, 0), 'angle': lambda x: 180 - abs(x), 'fn': lambda x: ((180 >= x >= 180-22.5) or (-180 <= x <= -180+22.5))}, 
                'top-left':     {'action': 2, 'signs': (-1, 1), 'angle': lambda x: abs(135 - x), 'fn': lambda x: (180-22.5 > x > 90+22.5)}, 
                'top':          {'action': 3, 'signs': (0, 1), 'angle': lambda x: abs(90 - x), 'fn': lambda x: (90+22.5 >= x >= 90-22.5)}, 
                'top-right':    {'action': 4, 'signs': (1, 1), 'angle': lambda x: abs(45 - x), 'fn': lambda x: (90-22.5 > x > 0+22.5)}, 
                'right':        {'action': 5, 'signs': (1, 0), 'angle': lambda x: abs(0 - x), 'fn': lambda x: (0+22.5 >= x >= 0-22.5)}, 
                'bottom-right': {'action': 6, 'signs': (1, -1), 'angle': lambda x: abs(-45 - x), 'fn': lambda x: (0-22.5 > x > -90+22.5)}, 
                'bottom':       {'action': 7, 'signs': (0, -1), 'angle': lambda x: abs(-90 - x), 'fn': lambda x: (-90+22.5 >= x >= -90-22.5)}, 
                'bottom-left':  {'action': 8, 'signs': (-1, -1), 'angle': lambda x: abs(-135 - x), 'fn': lambda x: (-90-22.5 > x > -180+22.5)},
} # divide into 8 cones

def precondition(obs):
    ball_owned_team = obs[0]['ball_owned_team']
    #print('ball owned team: ', ball_owned_team)
    if ball_owned_team == 0:
        return True
    return False

def postcondition(obs):
    return precondition(obs)

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def angle(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

def angle_to_direction(angle_radians):
    angle_degrees = angle_radians * 180.0 / math.pi
    angle_degrees = max(-180, min(180, angle_degrees))
    for direction in action_map.keys():
        if action_map[direction]['fn'](angle_degrees):
            return direction

def get_list_of_actions(action, acting_player):
    team_actions = [0] * N_PLAYERS_LEFT + [1] * N_PLAYERS_RIGHT
    team_actions[acting_player] = action
    return team_actions

def shortest_intercept_dist(kick_angle, dist_from_ball_to_reciever):
    delta_angle = action_map[angle_to_direction(kick_angle)]['angle'](kick_angle * 180.0 / math.pi)
    return abs(math.cos(delta_angle) * dist_from_ball_to_reciever)

def ball_pos_in_future(obs, prev_obs, kick_angle, dist_from_ball_to_reciever):
    x_ball, y_ball, z_ball = obs[0]['ball']
    x_prev_ball, y_prev_ball, z_prev_ball = prev_obs[0]['ball']
    v_x_sign = np.sign(x_ball - x_prev_ball)
    v_y_sign = np.sign(y_ball - y_prev_ball)
    
    x_sign, y_sign = action_map[angle_to_direction(kick_angle)]['signs']
    #print(f'hardcoded_signs {x_sign}, {y_sign}, signs: {v_x_sign}, {v_y_sign}') # check if 0 signs happen when angle that way

    SID = shortest_intercept_dist(kick_angle, dist_from_ball_to_reciever)

    x_ball_future = x_ball + x_sign*SID/np.sqrt(2)
    y_ball_future = x_ball + y_sign*SID/np.sqrt(2)
    return x_ball_future, y_ball_future

def track_ball_position(obs, x_ball, y_ball, reciever_num):
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    angle_radians = angle(x_ball, y_ball, x_rec, y_rec)
    action = action_map[angle_to_direction(angle_radians)]['action']
    action_list = get_list_of_actions(action, reciever_num)
    return action_list

def edge_conditions(obs, reciever_num):
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    # check if close to own goal
    distance_to_own_goal = dist(-1, 0, x_rec, y_rec)
    if distance_to_own_goal <= 0.2:
        print()
        return False
    # check if close to out line
    distance_to_out_line = min([abs(y_rec - (-0.42)), abs((0.42) - y_rec), abs(x_rec - (-1))])
    if distance_to_out_line <= 0.2:
        return False
    return True


def identify_free_player(obs):
    ball_owned_player = obs[0]['ball_owned_player'] # can be -1 if noone owns ball; have to check precondition before
    x_current, y_current = obs[0]['left_team'][ball_owned_player]
    x_ball, y_ball, z_ball = obs[0]['ball']

    # find freest players by treating opponents as recievers and comparing shortest intercept distances of both when in same cone
    free_players = {}
    # Also identify teammates far away who have few opponents around them
    high_pass_free_players = {}

    for i, player_coods in enumerate(obs[0]['left_team']):
        
        x, y = player_coods
        dist_from_ball_to_teammate = dist(x_current, y_current, x, y) 
        if dist_from_ball_to_teammate == 0:
            continue

        kick_angle_teammate = angle(x_ball, y_ball, x, y)
        SID_teammate = shortest_intercept_dist(kick_angle_teammate, dist_from_ball_to_teammate)
        
        num_opponents_that_can_intercept = 0
        num_opponents_around = 0

        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            dist_from_ball_to_opponent = dist(x_ball, y_ball, x_op, y_op)
            kick_angle_opponent = angle(x_ball, y_ball, x_op, y_op)
            if angle_to_direction(kick_angle_teammate) == angle_to_direction(kick_angle_opponent): # same cone
                SID_opponent = shortest_intercept_dist(kick_angle_opponent, dist_from_ball_to_opponent)
                if SID_opponent < SID_teammate: # compare SID's
                    #print(SID_opponent, SID_teammate)
                    num_opponents_that_can_intercept += 1

                dist_from_teammate_to_opponent = dist(x, y, x_op, y_op)
                if dist_from_teammate_to_opponent <= DIST_TO_OPP_THRESH and dist_from_ball_to_teammate > HIGH_THRESH: # check if opponent close, and if reciever further than threshold
                    num_opponents_around += 1     
        
        if num_opponents_that_can_intercept == 0 and edge_conditions(obs, i):
            free_players[i] = dist_from_ball_to_teammate

        if edge_conditions(obs, i):
            high_pass_free_players[i] = num_opponents_around


    # choose pass based on free player availability
    kick_type_map = {'short_pass': {'action': 11, 'cdn': lambda r: (SHORT_LONG_THRESH/2.0 <= r <= SHORT_LONG_THRESH), 'free_players': []}, 
                        'long_pass': {'action': 9, 'cdn': lambda r: (SHORT_LONG_THRESH <= r), 'free_players': []}, 
                        #'high_pass': {'action': 10, 'cdn': lambda r: (0.8 <= r), 'free_players': []}
                        }
    
    max_count = 0
    for kick_type in ['short_pass', 'long_pass']:
        for i, r in free_players.items():
            if kick_type_map[kick_type]['cdn']:
                kick_type_map[kick_type]['free_players'].append(i)
        
        print(kick_type, kick_type_map[kick_type]['free_players'])

        if len(kick_type_map[kick_type]['free_players']) > max_count:
            max_count = len(kick_type_map[kick_type]['free_players'])
            chosen_kick_type = kick_type

    if max_count == 0: # no available players for short or long pass
        print('Have to execute high pass! (no option)', high_pass_free_players)
        free_player = min(high_pass_free_players, key=high_pass_free_players.get)
        action_to_execute = 10
    else:
        free_player = random.choice(kick_type_map[chosen_kick_type]['free_players'])
        action_to_execute = kick_type_map[chosen_kick_type]['action']
    
    return free_player, action_to_execute
    
def pass_to_free_player(obs):

    ball_owned_player = obs[0]['ball_owned_player']
    x_current, y_current = obs[0]['left_team'][ball_owned_player]

    reciever_num, kick_type = identify_free_player(obs)

    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    kick_angle = angle(x_current, y_current, x_rec, y_rec)
    kick_dirn_action = action_map[angle_to_direction(kick_angle)]['action']
    
    kicker_actions = [get_list_of_actions(kick_dirn_action, ball_owned_player),
                        get_list_of_actions(kick_type, ball_owned_player)]

    return kicker_actions, reciever_num, kick_angle


if __name__ == "__main__":


    while True:
        obs = env.reset()
        done = False; i = 0
        while not done:
            if precondition(obs) or i == 0:
                actions, reciever_num, kick_angle = pass_to_free_player(obs)
                for action in actions:
                    prev_obs = obs
                    obs, reward, done, info = env.step(action)
                    i += 1
                while not postcondition(obs) and not done:
                    x_ball, y_ball, z_ball = obs[0]['ball']
                    x_rec, y_rec = obs[0]['left_team'][reciever_num]
                    dist_to_ball = dist(x_ball, y_ball, x_rec, y_rec)
                    if dist_to_ball > DIST_TO_BALL_THRESH:
                        x_ball_future, y_ball_future = ball_pos_in_future(obs, prev_obs, kick_angle, dist_to_ball)
                        action = track_ball_position(obs, x_ball_future, y_ball_future, reciever_num)
                        prev_obs = obs
                        obs, reward, done, info = env.step(action)
                        i += 1
                    else:
                        x_ball, y_ball, z_ball = obs[0]['ball']
                        action = track_ball_position(obs, x_ball, y_ball, reciever_num)
                        prev_obs = obs
                        obs, reward, done, info = env.step(action)
                        i += 1
            else:
                action = env.action_space.sample()
                prev_obs = obs
                obs, reward, done, info = env.step(action)
                i += 1