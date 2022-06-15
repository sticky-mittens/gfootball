import gfootball.env as football_env
import random
import numpy as np
import torch
import math

action_map = {'left':           {'action': 1, 'angle': lambda x: 180 - abs(x), 'fn': lambda x: ((180 >= x >= 180-22.5) or (-180 <= x <= -180+22.5))}, 
                'top-left':     {'action': 2, 'angle': lambda x: abs(135 - x), 'fn': lambda x: (180-22.5 > x > 90+22.5)}, 
                'top':          {'action': 3, 'angle': lambda x: abs(90 - x), 'fn': lambda x: (90+22.5 >= x >= 90-22.5)}, 
                'top-right':    {'action': 4, 'angle': lambda x: abs(45 - x), 'fn': lambda x: (90-22.5 > x > 0+22.5)}, 
                'right':        {'action': 5, 'angle': lambda x: abs(0 - x), 'fn': lambda x: (0+22.5 >= x >= 0-22.5)}, 
                'bottom-right': {'action': 6, 'angle': lambda x: abs(-45 - x), 'fn': lambda x: (0-22.5 > x > -90+22.5)}, 
                'bottom':       {'action': 7, 'angle': lambda x: abs(-90 - x), 'fn': lambda x: (-90+22.5 >= x >= -90-22.5)}, 
                'bottom-left':  {'action': 8, 'angle': lambda x: abs(-135 - x), 'fn': lambda x: (-90-22.5 > x > -180+22.5)},
} # divide into 8 cones

N_PLAYERS = 11 # change according to scenario below
N_PLAYERS_RIGHT = 11
dist_to_ball_threshold = 0.1
dist_to_opp_threshold = 0.1
r_thresh = 0.4

env = football_env.create_environment(
    env_name='11_vs_11_easy_stochastic',#'academy_3_vs_1_with_keeper',#'11_vs_11_easy_stochastic',#
    representation='raw',#'simple115v2',
    rewards='checkpoints,scoring',
    #write_full_episode_dumps=True,
    #write_video=True,
    #logdir='logs_opp/',
    number_of_left_players_agent_controls=N_PLAYERS,
    number_of_right_players_agent_controls=N_PLAYERS_RIGHT,
    render=True,
    other_config_options={'physics_steps_per_frame': 4},
    )

# def precondition(obs):
#     onehot_ball_own = obs[0, 94:97]
#     if torch.argmax(onehot_ball_own) == 1: # if left team (our team) owns ball
#         return True
#     return False

def precondition(obs):
    ball_owned_team = obs[0]['ball_owned_team']
    #print('ball owned team: ', ball_owned_team)
    if ball_owned_team == 0:
        return True
    return False

def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def angle(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

def identify_free_player_dirn(obs, r1, r2):
    #left_pos, left_dir, right_pos, right_dir, ball_pos, ball_dir, onehot_ball_own, onehot_players_active, onehot_game_mode = obs[0, 0:22], obs[0, 22:44], obs[0, 44:66], obs[0, 66:88], obs[0, 88:91], obs[0, 91:94], obs[0, 94:97], obs[0, 97:108], obs[0, 108:115]
    ball_owned_player = obs[0]['ball_owned_player'] # can be -1 if noone owns ball; have to check precondition before
    x_current, y_current = obs[0]['left_team'][ball_owned_player]

    # get all players in r1 < r < r2, save their direction of kicks too
    available_players = {}
    for i, player_coods in enumerate(obs[0]['left_team']):
        x, y = player_coods
        r = dist(x_current, y_current, x, y) 
        angle_radians = angle(x_current, y_current, x, y)
        direction = get_dir(angle_radians)
        distance_to_goal = dist(1, 0, x, y)
        if (r2 != -1 and r1 <= r <= r2) or (r2==-1 and r1 <= r):
            available_players[i] = {'x': x, 'y': y, 'direction': direction, 'goal_dist': distance_to_goal, 'nearby_opps': []}
    
    # get subset of available players with least opponents in same kicking direction (from kicker to player)
    n_opps = {}
    for i in available_players.keys():
        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            #d = dist(available_players[i]['x'], available_players[i]['y'], x_op, y_op)
            #if d_min <= d:
            #    available_players[i]['nearby_opps'].append(j)
            angle_radians_opp = angle(x_current, y_current, x_op, y_op)
            direction_opp = get_dir(angle_radians_opp)
            if direction_opp == available_players[i]['direction']:
                available_players[i]['nearby_opps'].append(j)
        n_opps[i] = len(available_players[i]['nearby_opps'])
    
    #print('number of opponents for each player is : {}'.format(n_opps))
    # find player with no opps closest to goal else return most free player
    zero_opp_idxs = []
    for idx in n_opps.keys():
        if n_opps[idx] == 0:
            zero_opp_idxs.append(idx)
    if len(zero_opp_idxs) > 0:
        return random.choice(zero_opp_idxs) # player with no opps closest to goal
    else:
        #print('LEAST CHOSEN!')
        return min(n_opps, key=n_opps.get) # equals argmin(n_opps) # most free player

def identify_free_player(obs, r1, r2, d_min = dist_to_opp_threshold):
    #left_pos, left_dir, right_pos, right_dir, ball_pos, ball_dir, onehot_ball_own, onehot_players_active, onehot_game_mode = obs[0, 0:22], obs[0, 22:44], obs[0, 44:66], obs[0, 66:88], obs[0, 88:91], obs[0, 91:94], obs[0, 94:97], obs[0, 97:108], obs[0, 108:115]
    ball_owned_player = obs[0]['ball_owned_player'] # can be -1 if noone owns ball; have to check precondition before
    x_current, y_current = obs[0]['left_team'][ball_owned_player]

    # get all players in r1 < r < r2, save their direction of kicks too
    available_players = {}
    for i, player_coods in enumerate(obs[0]['left_team']):
        x, y = player_coods
        r = dist(x_current, y_current, x, y) 
        angle_radians = angle(x_current, y_current, x, y)
        direction = get_dir(angle_radians)
        distance_to_goal = dist(1, 0, x, y)
        if (r2 != -1 and r1 <= r <= r2) or (r2==-1 and r1 <= r):
            available_players[i] = {'x': x, 'y': y, 'direction': direction, 'goal_dist': distance_to_goal, 'nearby_opps': [], 'dirn_opps': [], 'dirn_superset_opps': []}

    # get subset of available players with least opponents around them
    n_opps = {}
    for i in available_players.keys():
        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            d = dist(available_players[i]['x'], available_players[i]['y'], x_op, y_op)
            if d <= d_min:
                available_players[i]['nearby_opps'].append(j)
        n_opps[i] = len(available_players[i]['nearby_opps'])

    zero_opp_idxs = []
    for idx in n_opps.keys():
        if n_opps[idx] == 0:
            zero_opp_idxs.append(idx)

    # get subset of above subset with least opponents in same kicking direction (from kicker to player) ################################# dist of opp, kicker should be less than that of rec, kicker
    n_opps_in_dirn = {}
    for i in zero_opp_idxs:
        for j, opponent_coods in enumerate(obs[0]['right_team']):
            # x_op, y_op = opponent_coods
            # angle_radians_opp = angle(x_current, y_current, x_op, y_op)
            # direction_opp = get_dir(angle_radians_opp)
            # if direction_opp == available_players[i]['direction']:
            #     available_players[i]['dirn_opps'].append(j)

            # treat oppos as recs
            x_op, y_op = opponent_coods
            
        n_opps_in_dirn[i] = len(available_players[i]['dirn_opps'])

    # Also get subset of available players with least opponents in same kicking direction (from kicker to player)
    n_opps_in_dirn_superset = {}
    for i in available_players.keys():
        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            angle_radians_opp = angle(x_current, y_current, x_op, y_op)
            direction_opp = get_dir(angle_radians_opp)
            if direction_opp == available_players[i]['direction']:
                available_players[i]['dirn_superset_opps'].append(j)
        n_opps_in_dirn_superset[i] = len(available_players[i]['dirn_superset_opps'])
    
    #print('number of opponents for each player is : {}'.format(n_opps))
    # find player with no opps closest to goal else return most free player
    if len(n_opps_in_dirn.keys()) > 0:
        print('OPTION 1')
        return min(n_opps_in_dirn, key=n_opps_in_dirn.get) # equals argmin(n_opps) # most free player
    else:
        print('OPTION 3')
        return min(n_opps_in_dirn_superset, key=n_opps_in_dirn_superset.get)

def get_list_of_actions(action, acting_player, n_players = N_PLAYERS):
    team_actions = [0] * n_players + [1] * N_PLAYERS_RIGHT
    team_actions[acting_player] = action
    return team_actions

def get_kick_angle(obs, ball_owned_player_idx, reciever_idx):
    x_kicker, y_kicker = obs[0]['left_team'][ball_owned_player_idx]
    x_rec, y_rec = obs[0]['left_team'][reciever_idx]
    angle_radians = math.atan2(y_rec - y_kicker, x_rec - x_kicker)
    return angle_radians


def short_pass_to_free_player(obs, radius_threshold):
    ball_owned_player = obs[0]['ball_owned_player']
    reciever = identify_free_player(obs, r1=0, r2=radius_threshold)

    kick_orientation = get_kick_angle(obs, ball_owned_player, reciever)
    direction_type = get_dir(kick_orientation)

    kick_type = 11
    kicker_actions = [get_list_of_actions(direction_type, ball_owned_player), 
                        get_list_of_actions(kick_type, ball_owned_player)]
    return kicker_actions, reciever, kick_orientation

def long_pass_to_free_player(obs, radius_threshold):
    ball_owned_player = obs[0]['ball_owned_player']
    reciever = identify_free_player(obs, r1=radius_threshold, r2=-1)

    kick_orientation = get_kick_angle(obs, ball_owned_player, reciever)
    direction_type = get_dir(kick_orientation)

    kick_type = 9
    kicker_actions = [get_list_of_actions(direction_type, ball_owned_player), 
                        get_list_of_actions(kick_type, ball_owned_player)]
    return kicker_actions, reciever, kick_orientation

def high_pass_to_free_player(obs, radius_threshold):
    ball_owned_player = obs[0]['ball_owned_player']
    reciever = identify_free_player(obs, r1=radius_threshold, r2=-1)

    kick_orientation = get_kick_angle(obs, ball_owned_player, reciever)
    direction_type = get_dir(kick_orientation)

    kick_type = 10
    kicker_actions = [get_list_of_actions(direction_type, ball_owned_player), 
                        get_list_of_actions(kick_type, ball_owned_player)]
    return kicker_actions, reciever, kick_orientation

def get_dir(angle_radians):
    angle_degrees = angle_radians * 180.0 / math.pi
    angle_degrees = max(-180, min(180, angle_degrees))
    #if angle_degrees == -180: angle_degrees = 180
    for direction in action_map.keys():
        if action_map[direction]['fn'](angle_degrees):
            return action_map[direction]['action']

def get_diff_angle_with_dir(angle_radians):
    angle_degrees = angle_radians * 180.0 / math.pi
    angle_degrees = max(-180, min(180, angle_degrees))
    for direction in action_map.keys():
        if action_map[direction]['fn'](angle_degrees):
            dir = direction
            break
    return action_map[dir]['angle'](angle_degrees)

    
def track_ball_position(obs, x_ball, y_ball, reciever):
    x_rec, y_rec = obs[0]['left_team'][reciever]
    angle_radians = math.atan2(y_ball - y_rec, x_ball - x_rec)
    a = get_dir(angle_radians)
    a_list = get_list_of_actions(a, reciever)
    return a_list

def track_ball(obs, reciever, i, done, timeout = 50):
    ball_coods = obs[0]['ball']
    x_ball, y_ball, z_ball = ball_coods
    x_rec, y_rec = obs[0]['left_team'][reciever]

    #gap_to_ball = dist(x_ball, y_ball, x_rec, y_rec)
    count = 0
    while not done:
        angle_radians = math.atan2(y_ball - y_rec, x_ball - x_rec)
        a = get_dir(angle_radians)
        a_list = get_list_of_actions(a, reciever)
        #print('tracker @ step {}: {}'.format(i, a_list))
        observation, reward, done, info = env.step(a_list)
        count += 1

        if postcondition(obs) or count>=timeout:
            #print('Got the ball')
            break

    return observation, reward, done, info, i+count

def get_dist_to_ball(obs, reciever):
    ball_coods = obs[0]['ball']
    x_ball, y_ball, z_ball = ball_coods
    x_rec, y_rec = obs[0]['left_team'][reciever]
    return dist(x_ball, y_ball, x_rec, y_rec)

def get_ball_pos_k_steps_in_future(obs, prev_obs, kick_angle, dist_ball_reciever):
    delta_angle_degrees = get_diff_angle_with_dir(kick_angle)
    delta_angle = delta_angle_degrees * math.pi/180.0

    x_ball, y_ball, z_ball = obs[0]['ball']
    x_prev_ball, y_prev_ball, z_prev_ball = prev_obs[0]['ball']

    #current_speed_of_ball = (y_ball - y_prev_ball)/(x_ball - x_prev_ball)
    v_x_sign = np.sign(x_ball - x_prev_ball)
    v_y_sign = np.sign(y_ball - y_prev_ball)
    length_after_K = math.cos(delta_angle) * dist_ball_reciever
    #return x_ball + length_after_K*1/math.sqrt(2), y_ball + length_after_K*1/math.sqrt(2) ########################################### need to change sign here ### vel * t
    return x_ball + v_x_sign*length_after_K*1/math.sqrt(2), y_ball + v_y_sign*length_after_K/math.sqrt(2)


postcondition = precondition
def main():
    while True:
        observation = env.reset()
        done = False; i = 0; saved_state = 0
        while not done:
            if precondition(observation) or i == 0:
                state = env.get_state()
                saved_state = 1
                actions, reciever, kick_orientation = high_pass_to_free_player(observation, radius_threshold=r_thresh)
                for a in actions:
                    #print('kicker @ step {}: {}'.format(i, a))
                    prev_observation = observation
                    observation, reward, done, info = env.step(a)
                    i += 1
                while not postcondition(observation) and not done:
                    dist_to_ball = get_dist_to_ball(observation, reciever)
                    if dist_to_ball > dist_to_ball_threshold:
                        x_ball_future, y_ball_future = get_ball_pos_k_steps_in_future(observation, prev_observation, kick_orientation, dist_to_ball)
                        a = track_ball_position(observation, x_ball_future, y_ball_future, reciever)
                        prev_observation = observation
                        observation, reward, done, info = env.step(a)
                        i += 1
                    else:
                        x_ball, y_ball, z_ball = observation[0]['ball']
                        a = track_ball_position(observation, x_ball, y_ball, reciever)
                        prev_observation = observation
                        observation, reward, done, info = env.step(a)
                        i += 1
            else:
                # if saved_state:
                #     env.set_state(state)
                #     print('reset state complete!')
                #     observation = env.observation()
                #     continue
                action = env.action_space.sample()
                #print('random: ', action)
                prev_observation = observation
                observation, reward, done, info = env.step(action)
                i += 1

        

        # print(i, len(observation), len(observation[0]), '\n')
        # # # print(observation, '\n')
        # # break
        # for i in range(len(observation)):
        #     print(observation[i]['ball_owned_player'])
        #     print(observation[0]['left_team'])
        # #break
        # i += 1


if __name__ == "__main__":
    main()