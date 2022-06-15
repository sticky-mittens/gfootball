import random
import numpy as np
import math 

DEBUG = False

########## HELPERS ##########################
def dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def angle(x1, y1, x2, y2):
    return math.atan2(y2-y1, x2-x1)

action_map = {'left':           {'action': 1, 'signs': (-1, 0), 'angle': lambda x: 180 - abs(x), 'fn': lambda x: ((180 >= x >= 180-22.5) or (-180 <= x <= -180+22.5))}, 
                'bottom-left':  {'action': 8, 'signs': (-1, 1), 'angle': lambda x: abs(135 - x), 'fn': lambda x: (180-22.5 > x > 90+22.5)}, 
                'top':          {'action': 3, 'signs': (0, 1), 'angle': lambda x: abs(90 - x), 'fn': lambda x: (90+22.5 >= x >= 90-22.5)}, 
                'bottom-right': {'action': 6, 'signs': (1, 1), 'angle': lambda x: abs(45 - x), 'fn': lambda x: (90-22.5 > x > 0+22.5)}, 
                'right':        {'action': 5, 'signs': (1, 0), 'angle': lambda x: abs(0 - x), 'fn': lambda x: (0+22.5 >= x >= 0-22.5)}, 
                'top-right':    {'action': 4, 'signs': (1, -1), 'angle': lambda x: abs(-45 - x), 'fn': lambda x: (0-22.5 > x > -90+22.5)}, 
                'bottom':       {'action': 7, 'signs': (0, -1), 'angle': lambda x: abs(-90 - x), 'fn': lambda x: (-90+22.5 >= x >= -90-22.5)}, 
                'top-left':     {'action': 2, 'signs': (-1, -1), 'angle': lambda x: abs(-135 - x), 'fn': lambda x: (-90-22.5 > x > -180+22.5)},
} # divide into 8 cones

def angle_to_direction(angle_radians):
    angle_degrees = angle_radians * 180.0 / math.pi
    old_angle_degrees = angle_degrees
    # #if DEBUG: print('pre boudning: ', angle_degrees)
    angle_degrees = max(-180, min(180, angle_degrees))
    # #if DEBUG: print('post: ', angle_degrees)
    # if old_angle_degrees != angle_degrees:
    #     exit(0)
    for direction in action_map.keys():
        if action_map[direction]['fn'](angle_degrees):
            # #if DEBUG: print('corr dir: ', direction)
            return direction

def angle_to_action(angle_radians):
    return action_map[angle_to_direction(angle_radians)]['action']

def get_action_for(player_tuple, point_to_reach_tuple):
    player_num, dist_to_own_goal, x_player, y_player = player_tuple
    px, py = point_to_reach_tuple
    angle_player = angle(x_player, y_player, px, py)
    action = angle_to_action(angle_player)
    return action

def check_on_defenders(obs, acting_player_num):
    players = []
    n_defenders_already = 0
    p1 = (-1 + DEFENDERS_THRESH/np.sqrt(2), 0 + DEFENDERS_THRESH/np.sqrt(2))
    p2 = (-1 + DEFENDERS_THRESH/np.sqrt(2), 0 - DEFENDERS_THRESH/np.sqrt(2))
    for i, player_coods in enumerate(obs[0]['left_team']):
        if not is_goalkeeper(obs, i) and i != acting_player_num: # i.e. if not goalkeeper or acting player
            x, y = player_coods
            dist_to_own_goal = dist(-1, 0, x, y)
            if dist_to_own_goal <= DEFENDERS_THRESH*1.25:
                n_defenders_already += 1
            players.append((i, dist_to_own_goal, x, y))

    players.sort(key = lambda x: x[1]) # sort by dist_to_own_goal
    
    if n_defenders_already >= 2:
        return [(True, players[0][0], None), (True, players[1][0], None)]

    # closest 2 players run to defend own goal
    action_2 = get_action_for(players[1], p2)

    if n_defenders_already == 1:
        return [(True, players[0][0], None), (False, players[1][0], action_2)]

    action_1 = get_action_for(players[0], p1)

    return [(False, players[0][0], action_1), (False, players[1][0], action_2)]

def check_on_wings(obs, acting_player_num, def_num_1, def_num_2, LOCAL_WINGS_THRESH):
    # x_act, y_act = obs[0]['left_team'][acting_player_num]
    players = []
    n_wings_already = 0
    p1 = (1 - LOCAL_WINGS_THRESH/np.sqrt(2), 0 + LOCAL_WINGS_THRESH/np.sqrt(2))
    p2 = (1 - LOCAL_WINGS_THRESH/np.sqrt(2), 0 - LOCAL_WINGS_THRESH/np.sqrt(2))
    for i, player_coods in enumerate(obs[0]['left_team']):
        if not is_goalkeeper(obs, i) and i not in [acting_player_num, def_num_1, def_num_2]: # i.e. if not goalkeeper or acting player/def_1/def_2
            x, y = player_coods
            dist_to_opp_goal = dist(1, 0, x, y)
            if dist_to_opp_goal <= LOCAL_WINGS_THRESH*1.25:
                n_wings_already += 1
            players.append((i, dist_to_opp_goal, x, y))

    players.sort(key = lambda x: x[1]) # sort by dist_to_opp_goal
    
    if n_wings_already >= 2:
        return [(True, players[0][0], None), (True, players[1][0], None)]
    
    # closest 2 players to opp goal run forward as wings
    action_2 = get_action_for(players[1], p2)

    if n_wings_already == 1:
        return [(True, players[0][0], None), (False, players[1][0], action_2)]

    action_1 = get_action_for(players[0], p1)

    return [(False, players[0][0], action_1), (False, players[1][0], action_2)]

def check_on_midfields(obs, acting_player_num, def_num_1, def_num_2):
    players = []
    n_retreated_already = 0
    m_bottom = (-1 + MIDFIELDS_RETREATING_THRESH/np.sqrt(2), 0 + MIDFIELDS_RETREATING_THRESH/np.sqrt(2))
    m_top = (-1 + MIDFIELDS_RETREATING_THRESH/np.sqrt(2), 0 - MIDFIELDS_RETREATING_THRESH/np.sqrt(2))
    m_center = (-1 + MIDFIELDS_RETREATING_THRESH/np.sqrt(2), 0)
    for i, player_coods in enumerate(obs[0]['left_team']):
        if not is_goalkeeper(obs, i) and i not in [acting_player_num, def_num_1, def_num_2]: # i.e. if not goalkeeper or acting player/def_1/def_2
            x, y = player_coods
            dist_to_own_goal = dist(-1, 0, x, y)
            if dist_to_own_goal <= MIDFIELDS_RETREATING_THRESH*1.5:
                n_retreated_already += 1
            players.append((i, dist_to_own_goal, x, y))

    players.sort(key = lambda x: x[1]) # sort by dist_to_own_goal
    
    if n_retreated_already >= 3:
        return [(True, players[0][0], None), (True, players[1][0], None), (True, players[2][0], None)]

    # closest 3 players (apart from goalie/acting player/def_1/def_2) run to defend own goal
    action_3 = get_action_for(players[2], m_center)

    if n_retreated_already == 2:
        return [(True, players[0][0], None), (True, players[1][0], None), (False, players[2][0], action_3)]

    action_2 = get_action_for(players[1], m_top)

    if n_retreated_already == 1:
        return [(True, players[0][0], None), (False, players[1][0], action_2), (False, players[2][0], action_3)]

    action_1 = get_action_for(players[0], m_bottom)

    return [(False, players[0][0], action_1), (False, players[1][0], action_2), (False, players[2][0], action_3)]

def update_team_actions(team_actions, list_of_tuples, resting_action):
    for i, tuple in enumerate(list_of_tuples):
        player_already_in_correct_position, player_num, player_action = tuple
        if not player_already_in_correct_position:
            team_actions[player_num] = player_action
        else:
            team_actions[player_num] = resting_action
    return team_actions

def set_goalkeeper_to_builtin_ai(teams_actions, obs):
    for i in range(11):
        if is_goalkeeper(obs, i):
            goalie = i
            break
    teams_actions[goalie] = 19
    return teams_actions

def get_list_of_actions(obs, action, acting_player):
    if is_goalkeeper(obs, acting_player):
        if DEBUG: print('ACTING PLAYER IS GOALKEEPER -------------------------------------------------------------------------!!!!!!!!!!!!!')
    if N_PLAYERS_LEFT == 1:
        return [action]
    team_actions = [BASE_ACTION] * N_PLAYERS_LEFT + [1] * N_PLAYERS_RIGHT
    team_actions[acting_player] = action

    list_of_def_tuples = check_on_defenders(obs, acting_player)
    team_actions = update_team_actions(team_actions, list_of_def_tuples, resting_action = 14)

    team_actions = set_goalkeeper_to_builtin_ai(team_actions, obs)
    # if DEBUG: print('get list of actions: ', team_actions)
    return team_actions

def get_list_of_actions_with_midfields_retreating(obs, action, acting_player):
    if is_goalkeeper(obs, acting_player):
        if DEBUG: print('ACTING PLAYER IS GOALKEEPER -------------------------------------------------------------------------!!!!!!!!!!!!!')
    if N_PLAYERS_LEFT == 1:
        return [action]
    team_actions = [BASE_ACTION] * N_PLAYERS_LEFT + [1] * N_PLAYERS_RIGHT
    team_actions[acting_player] = action

    list_of_def_tuples = check_on_defenders(obs, acting_player)
    team_actions = update_team_actions(team_actions, list_of_def_tuples, resting_action = 14)
    
    _, def_num_1, _ = list_of_def_tuples[0]
    _, def_num_2, _ = list_of_def_tuples[1]

    list_of_midfield_tuples = check_on_midfields(obs, acting_player, def_num_1, def_num_2)
    team_actions = update_team_actions(team_actions, list_of_midfield_tuples, resting_action = 14)

    # move wings back to WINGS_THRESH
    list_of_wing_tuples = check_on_wings(obs, acting_player, def_num_1, def_num_2, WINGS_THRESH)
    team_actions = update_team_actions(team_actions, list_of_wing_tuples, resting_action = 14)

    team_actions = set_goalkeeper_to_builtin_ai(team_actions, obs)
    return team_actions

def get_list_of_actions_with_wings(obs, action, acting_player):
    if is_goalkeeper(obs, acting_player):
        if DEBUG: print('ACTING PLAYER IS GOALKEEPER -------------------------------------------------------------------------!!!!!!!!!!!!!')
    if N_PLAYERS_LEFT == 1:
        return [action]
    team_actions = [BASE_ACTION] * N_PLAYERS_LEFT + [1] * N_PLAYERS_RIGHT
    team_actions[acting_player] = action

    list_of_def_tuples = check_on_defenders(obs, acting_player)
    team_actions = update_team_actions(team_actions, list_of_def_tuples, resting_action = 14)
    
    _, def_num_1, _ = list_of_def_tuples[0]
    _, def_num_2, _ = list_of_def_tuples[1]

    # move wings progressively forward starting from WINGS_THRESH to ahead of forwardmost player
    x_actor, y_actor = obs[0]['left_team'][acting_player]
    if WINGS_THRESH > 1 - x_actor:
        LOCAL_WINGS_THRESH = (1 - x_actor) - WINGS_DIFF_WITH_FRONTMOST_PLAYER_THRESH
    else:
        LOCAL_WINGS_THRESH = WINGS_THRESH
    list_of_wing_tuples = check_on_wings(obs, acting_player, def_num_1, def_num_2, LOCAL_WINGS_THRESH)
    team_actions = update_team_actions(team_actions, list_of_wing_tuples, resting_action = 14)

    team_actions = set_goalkeeper_to_builtin_ai(team_actions, obs)
    return team_actions

# def get_list_of_actions_multi_actors(action_agent_tuples):
#     team_actions = [14] * N_PLAYERS_LEFT + [1] * N_PLAYERS_RIGHT
#     for idx, (action, agent) in enumerate(action_agent_tuples):
#         team_actions[agent] = action
#     return team_actions

def shortest_intercept_dist(kick_angle, dist_from_ball_to_reciever):
    delta_angle = action_map[angle_to_direction(kick_angle)]['angle'](kick_angle * 180.0 / math.pi)
    return abs(math.cos(delta_angle) * dist_from_ball_to_reciever)

def ball_pos_in_future(obs, kick_angle, dist_from_ball_to_reciever):
    x_ball, y_ball, z_ball = obs[0]['ball']
    # x_prev_ball, y_prev_ball, z_prev_ball = prev_obs[0]['ball']
    # v_x_sign = np.sign(x_ball - x_prev_ball)
    # v_y_sign = np.sign(y_ball - y_prev_ball)
    
    x_sign, y_sign = action_map[angle_to_direction(kick_angle)]['signs']
    ##if DEBUG: print(f'hardcoded_signs {x_sign}, {y_sign}, signs: {v_x_sign}, {v_y_sign}') # check if 0 signs happen when angle that way

    SID = shortest_intercept_dist(kick_angle, dist_from_ball_to_reciever)

    x_ball_future = x_ball + x_sign*SID/np.sqrt(2)
    y_ball_future = x_ball + y_sign*SID/np.sqrt(2)
    return x_ball_future, y_ball_future

def track_position(obs, x_ball, y_ball, reciever_num, extra_action = None): # sprint: 13 (release: 15), dribble: 17 (release: 18)
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    angle_radians = angle(x_rec, y_rec, x_ball, y_ball)
    action = angle_to_action(angle_radians)
    action_list = [get_list_of_actions(obs, action, reciever_num)]
    if extra_action is not None:
        action_list.append(get_list_of_actions(obs, extra_action, reciever_num))
    return action_list

def track_position_in_take_ball(obs, x_ball, y_ball, reciever_num, extra_action = None): # sprint: 13 (release: 15), dribble: 17 (release: 18)
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    angle_radians = angle(x_rec, y_rec, x_ball, y_ball)
    action = angle_to_action(angle_radians)
    action_list = [get_list_of_actions_with_midfields_retreating(obs, action, reciever_num)]
    if extra_action is not None:
        action_list.append(get_list_of_actions_with_midfields_retreating(obs, extra_action, reciever_num))
    return action_list

def isOffside(obs, reciever_num):
    # offside if player is to right of ball and right of second last opponent
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    x_ball, y_ball, z_ball = obs[0]['ball']

    isRightOfBall = x_rec > x_ball

    n_opps_to_right_of_rec = 0
    for j, opponent_coods in enumerate(obs[0]['right_team']):
        x_op, y_op = opponent_coods
        if x_op > x_rec:
            n_opps_to_right_of_rec += 1

    if isRightOfBall and n_opps_to_right_of_rec <= 1:
        return True
    
    return False

def edge_conditions_with_offside(obs, reciever_num, edge_thresh = 0.2):
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    # check if close to own goal
    distance_to_own_goal = dist(-1, 0, x_rec, y_rec)
    if distance_to_own_goal <= edge_thresh:
        return False
    # check if close to out line
    distance_to_out_line = min([abs(y_rec - (-0.42)), abs((0.42) - y_rec), abs(x_rec - (-1)), abs(x_rec - (1))])
    if distance_to_out_line <= edge_thresh:
        return False

    # check for offside
    if isOffside(obs, reciever_num):
        return False
    
    return True

def edge_conditions(obs, reciever_num, edge_thresh = 0.2):
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    # check if close to own goal
    distance_to_own_goal = dist(-1, 0, x_rec, y_rec)
    if distance_to_own_goal <= edge_thresh:
        return False
    # check if close to out line
    distance_to_out_line = min([abs(y_rec - (-0.42)), abs((0.42) - y_rec), abs(x_rec - (-1)), abs(x_rec - (1))])
    if distance_to_out_line <= edge_thresh:
        return False
    return True

def identify_free_player(obs):
    ball_owned_player = obs[0]['ball_owned_player'] # can be -1 if noone owns ball; have to check precondition before
    x_current, y_current = obs[0]['left_team'][ball_owned_player]
    x_ball, y_ball, z_ball = obs[0]['ball']

    # find freest players by treating opponents as recievers and comparing shortest intercept distances of both when in same cone
    short_pass_free_players = []
    long_pass_free_players = []
    # Also identify teammates far away who have few opponents around them
    high_pass_free_players = []
    pass_action_map = {'short': 11, 'long': 9, 'high': 10}
    reverse_pass_action_map = {11: 'short', 9: 'long', 10: 'high'}

    for i, player_coods in enumerate(obs[0]['left_team']):
        
        x, y = player_coods
        dist_from_ball_to_teammate = dist(x_ball, y_ball, x, y) 
        if x == x_current and y == y_current:
            continue

        kick_angle_teammate = angle(x_ball, y_ball, x, y)
        SID_teammate = shortest_intercept_dist(kick_angle_teammate, dist_from_ball_to_teammate)
        
        num_opponents_that_can_intercept = 0
        num_opponents_around = 0

        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            kick_angle_opponent = angle(x_ball, y_ball, x_op, y_op)
            if angle_to_direction(kick_angle_teammate) == angle_to_direction(kick_angle_opponent): # if same cone
                dist_from_ball_to_opponent = dist(x_ball, y_ball, x_op, y_op)
                SID_opponent = shortest_intercept_dist(kick_angle_opponent, dist_from_ball_to_opponent)
                if SID_opponent <= SID_teammate: # compare SID's
                    num_opponents_that_can_intercept += 1

                dist_from_teammate_to_opponent = dist(x, y, x_op, y_op)
                if dist_from_teammate_to_opponent <= DIST_TO_OPP_THRESH: # check if opponent close
                    num_opponents_around += 1     
        
        # if edge_conditions_with_offside(obs, i):
        if edge_conditions(obs, i):
            dist_from_teammate_to_opp_goal = dist(1, 0, x, y)
            if num_opponents_that_can_intercept == 0: # no opponent interceptors
                if SHORT_MIN_THRESH <= dist_from_ball_to_teammate <= SHORT_LONG_THRESH:
                    short_pass_free_players.append((i, dist_from_ball_to_teammate, dist_from_teammate_to_opp_goal, pass_action_map['short']))
                elif SHORT_LONG_THRESH <= dist_from_ball_to_teammate:
                    long_pass_free_players.append((i, dist_from_ball_to_teammate, dist_from_teammate_to_opp_goal, pass_action_map['long']))
            else: # if there are opponent interceptors, do high pass to faraway teammate
                if HIGH_THRESH <= dist_from_ball_to_teammate:
                    high_pass_free_players.append((i, num_opponents_around, dist_from_teammate_to_opp_goal, pass_action_map['high']))

    all_short_long_free_players = short_pass_free_players + long_pass_free_players
    if len(all_short_long_free_players) > 0:

        all_short_long_free_players.sort(key = lambda x: x[2]) # sort by dist to opp goal
        best_free_player = all_short_long_free_players[0][0]
        action_to_execute = all_short_long_free_players[0][3]
        #if DEBUG: print('Performing {} pass (1st/2nd pref) to free player {}'.format(reverse_pass_action_map[action_to_execute], best_free_player))

    elif len(high_pass_free_players) > 0:

        high_pass_free_players.sort(key = lambda x: x[1]) # sort by number of opponents around
        best_free_player = high_pass_free_players[0][0]
        #if DEBUG: print(f'Performing HIGH pass (3rd pref) to free player {best_free_player}')
        action_to_execute = 10

    else:
        # pass to some random player
        #if DEBUG: print('No free players ==> passing to random player (no other option)!') # using distance to random player to figure out what pass action
        all_players_but_current = [i for i in range(ball_owned_player)] + [j for j in range(ball_owned_player+1, 11)]
        best_free_player = random.choice(all_players_but_current)
        x_, y_ = obs[0]['left_team'][best_free_player]
        r_ = dist(x_current, y_current, x_, y_)
        
        if r_ <= SHORT_LONG_THRESH:
            action_to_execute = 11
        elif SHORT_LONG_THRESH <= r_ <= HIGH_THRESH:
            action_to_execute = 9
        else:
            action_to_execute = 10
    
    return best_free_player, action_to_execute

def identify_free_choice(obs, choice_num):
    ball_owned_player = obs[0]['ball_owned_player'] # can be -1 if noone owns ball; have to check precondition before
    x_current, y_current = obs[0]['left_team'][ball_owned_player]
    x_ball, y_ball, z_ball = obs[0]['ball']

    # find freest players by treating opponents as recievers and comparing shortest intercept distances of both when in same cone
    short_pass_free_players = []
    long_pass_free_players = []
    # Also identify teammates far away who have few opponents around them
    high_pass_free_players = []
    pass_action_map = {'short': 11, 'long': 9, 'high': 10}
    reverse_pass_action_map = {11: 'short', 9: 'long', 10: 'high'}

    for i, player_coods in enumerate(obs[0]['left_team']):
        
        x, y = player_coods
        dist_from_ball_to_teammate = dist(x_ball, y_ball, x, y) 
        if x == x_current and y == y_current:
            continue

        kick_angle_teammate = angle(x_ball, y_ball, x, y)
        SID_teammate = shortest_intercept_dist(kick_angle_teammate, dist_from_ball_to_teammate)
        
        num_opponents_that_can_intercept = 0
        num_opponents_around = 0

        for j, opponent_coods in enumerate(obs[0]['right_team']):
            x_op, y_op = opponent_coods
            kick_angle_opponent = angle(x_ball, y_ball, x_op, y_op)
            if angle_to_direction(kick_angle_teammate) == angle_to_direction(kick_angle_opponent): # if same cone
                dist_from_ball_to_opponent = dist(x_ball, y_ball, x_op, y_op)
                SID_opponent = shortest_intercept_dist(kick_angle_opponent, dist_from_ball_to_opponent)
                if SID_opponent <= SID_teammate: # compare SID's
                    num_opponents_that_can_intercept += 1

                dist_from_teammate_to_opponent = dist(x, y, x_op, y_op)
                if dist_from_teammate_to_opponent <= RELAXED_DIST_TO_OPP_THRESH: # check if opponent close
                    num_opponents_around += 1     
        
        # if edge_conditions_with_offside(obs, i):
        if edge_conditions(obs, i):
            dist_from_teammate_to_opp_goal = dist(1, 0, x, y)
            if num_opponents_that_can_intercept == 0: # no opponent interceptors
                if RELAXED_SHORT_MIN_THRESH <= dist_from_ball_to_teammate <= RELAXED_SHORT_LONG_THRESH:
                    short_pass_free_players.append((i, dist_from_ball_to_teammate, dist_from_teammate_to_opp_goal, pass_action_map['short']))
                elif RELAXED_SHORT_LONG_THRESH <= dist_from_ball_to_teammate:
                    long_pass_free_players.append((i, dist_from_ball_to_teammate, dist_from_teammate_to_opp_goal, pass_action_map['long']))
            else: # if there are opponent interceptors, do high pass to faraway teammate
                if RELAXED_HIGH_THRESH <= dist_from_ball_to_teammate:
                    high_pass_free_players.append((i, num_opponents_around, dist_from_teammate_to_opp_goal, pass_action_map['high']))

    all_free_players = short_pass_free_players + long_pass_free_players + high_pass_free_players
    if SHUFFLE_IN_IDENTIFY_FREE_CHOICE:
        random.shuffle(all_free_players)

    if len(all_free_players) > 0:
        chosen_player, _, action_to_execute = all_free_players[choice_num-1] # chosen free player, _, action_to_execute

    else:
        all_players_but_current = [i for i in range(ball_owned_player)] + [j for j in range(ball_owned_player+1, 11)]
        chosen_player = all_players_but_current[choice_num-1] # chosen random player

        x_, y_ = obs[0]['left_team'][chosen_player]
        r_ = dist(x_current, y_current, x_, y_)
        
        if r_ <= RELAXED_SHORT_LONG_THRESH:
            action_to_execute = 11
        elif RELAXED_SHORT_LONG_THRESH <= r_ <= RELAXED_HIGH_THRESH:
            action_to_execute = 9
        else:
            action_to_execute = 10
    
    return chosen_player, action_to_execute

def pass_to_free_player(obs):

    ball_owned_player = obs[0]['ball_owned_player']
    x_current, y_current = obs[0]['left_team'][ball_owned_player]

    reciever_num, kick_type = identify_free_player(obs)

    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    kick_angle = angle(x_current, y_current, x_rec, y_rec)
    kick_dirn_action = angle_to_action(kick_angle)
    
    kicker_actions = [get_list_of_actions(obs, kick_dirn_action, ball_owned_player),
                        get_list_of_actions(obs, kick_type, ball_owned_player)]

    return kicker_actions, reciever_num, kick_angle

def pass_to_free_choice(obs, choice_num):

    ball_owned_player = obs[0]['ball_owned_player']
    x_current, y_current = obs[0]['left_team'][ball_owned_player]

    reciever_num, kick_type = identify_free_choice(obs, choice_num)

    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    kick_angle = angle(x_current, y_current, x_rec, y_rec)
    kick_dirn_action = angle_to_action(kick_angle)
    
    kicker_actions = [get_list_of_actions(obs, kick_dirn_action, ball_owned_player),
                        get_list_of_actions(obs, kick_type, ball_owned_player)]

    return kicker_actions, reciever_num, kick_angle
    
def dribble_towards_goal(obs, run_dirn):
    ball_owned_player = obs[0]['ball_owned_player']

    run_dirn_action = action_map[run_dirn]['action']

    dribble_action = 17
    actions = [get_list_of_actions(obs, run_dirn_action, ball_owned_player)]#,
                        #get_list_of_actions(obs, dribble_action, ball_owned_player)]
    return actions ###################### when to release dribble?

def dribble_towards_goal_with_wings(obs, run_dirn):
    ball_owned_player = obs[0]['ball_owned_player']

    run_dirn_action = action_map[run_dirn]['action']

    dribble_action = 17
    actions = [get_list_of_actions_with_wings(obs, run_dirn_action, ball_owned_player)]#,
                        #get_list_of_actions(obs, dribble_action, ball_owned_player)]
    return actions 

def identify_free_space(obs, ct, constrained_to_goal_directions = False, choose_from_all_directions = False):

    ball_owned_player = obs[0]['ball_owned_player']     
    x, y = obs[0]['left_team'][ball_owned_player]
    
    # dist_to_goal = dist(x, y, 1, 0)
    dirn_to_goal_center = angle_to_direction(angle(x, y, 1, 0))
    dirn_to_goal_top = angle_to_direction(angle(x, y, 1, -0.044))
    dirn_to_goal_bottom = angle_to_direction(angle(x, y, 1, 0.044))

    if not constrained_to_goal_directions: # OG: ct<=50
        if ct<=10 or choose_from_all_directions: # in the first few (50) steps, during kickoff, you should be able to move in any direction to just not lose ball
            num_opponents_around = {'right': 0, 'top-right': 0, 'bottom-right': 0, 'top': 0, 'bottom': 0, 'top-left': 0, 'bottom-left': 0, 'left': 0}
        else:
            num_opponents_around = {'right': 0, 'top-right': 0, 'bottom-right': 0, 'top': 0, 'bottom': 0}
    else:
        num_opponents_around = {}
        if y <= -0.044: # if above goal
            constr_directions = [dirn_to_goal_center, dirn_to_goal_bottom]
        elif y >= 0.044: # if below goal
            constr_directions = [dirn_to_goal_center, dirn_to_goal_top]
        else:
            constr_directions = [dirn_to_goal_center, dirn_to_goal_top, dirn_to_goal_bottom]
        #if DEBUG: print('The constrained directions are: ', constr_directions)
        for dir in constr_directions:
            num_opponents_around[dir] = 0

    for j, opponent_coods in enumerate(obs[0]['right_team']):
        if not is_opp_goalkeeper(obs, j): # opponent goalkeeper not counted
            x_op, y_op = opponent_coods
            dist_from_player_to_opponent = dist(x, y, x_op, y_op)

            is_opponent_on_right = (x_op >= x)
            is_opponent_nearby = (dist_from_player_to_opponent <= DIST_TO_OPP_THRESH) # prev: no relaxed
            if is_opponent_on_right and is_opponent_nearby:
                angle_opponent = angle(x, y, x_op, y_op)
                direction = angle_to_direction(angle_opponent)
                if direction in num_opponents_around:
                    num_opponents_around[direction] += 1
    
    for key, num_opp in num_opponents_around.items():
        # if dist_to_goal <= 1.5*GOAL_THRESH and key != dirn_to_goal:
        #     continue
        if num_opp == 0 and edge_conditions(obs, ball_owned_player, edge_thresh=0.05):
            return key, True
    return None, False


def identify_best_intercepter(obs, obs_prev, metric): # metric in ['SID', 'closest']
    x_ball_prev, y_ball_prev, z_ball_prev = obs_prev[0]['ball']
    x_ball, y_ball, z_ball = obs[0]['ball']

    interceptors_on_the_left = []
    other_interceptors = []
    for i, player_coods in enumerate(obs[0]['left_team']):
        
        if not is_goalkeeper(obs, i):
            x, y = player_coods
            run_angle_opponent = angle(x_ball_prev, y_ball_prev, x_ball, y_ball)
            dist_from_ball_to_interceptor = dist(x_ball, y_ball, x, y)
            SID_interceptor = shortest_intercept_dist(run_angle_opponent, dist_from_ball_to_interceptor)

            if x <= x_ball:
                interceptors_on_the_left.append((i, dist_from_ball_to_interceptor, SID_interceptor))
            else:
                other_interceptors.append((i, dist_from_ball_to_interceptor, SID_interceptor))

    if len(interceptors_on_the_left) > 0: 
        if metric == 'SID':
            interceptors_on_the_left.sort(key = lambda x: x[2]) # sort list of 3-value-tuples by third value (SID)
        else: # metric == 'closest'
            interceptors_on_the_left.sort(key = lambda x: x[1]) # sort list of 3-value-tuples by 2nd value (closest distance)
        best_interceptor_num = interceptors_on_the_left[0][0] # return shortest SID interceptor
    else:
        if metric == 'SID':
            other_interceptors.sort(key = lambda x: x[2]) # sort list by SID
        else: # metric == 'closest'
            other_interceptors.sort(key = lambda x: x[1]) # sort list by closest distance
        best_interceptor_num = other_interceptors[0][0] # return shortest SID interceptor

    
    return best_interceptor_num, is_goalkeeper(obs, best_interceptor_num)

#############################################
# MAIN FNS
#############################################

def have_ball(obs):
    # x_ball, y_ball, z_ball = obs[0]['ball']
    #if DEBUG: print('ball pos: (', x_ball, ', ', y_ball, ')')

    ball_owned_team = obs[0]['ball_owned_team']
    if ball_owned_team == 0:
        return True
    return False

def not_have_ball(obs):
    return not have_ball(obs)

def no_one_has_ball(obs):
    # x_ball, y_ball, z_ball = obs[0]['ball']
    #if DEBUG: print('ball pos: (', x_ball, ', ', y_ball, ')')

    ball_owned_team = obs[0]['ball_owned_team']
    if ball_owned_team == -1:
        return True
    return False

def opponents_have_ball(obs):
    # x_ball, y_ball, z_ball = obs[0]['ball']
    #if DEBUG: print('ball pos: (', x_ball, ', ', y_ball, ')')

    ball_owned_team = obs[0]['ball_owned_team']
    if ball_owned_team == 1:
        return True
    return False

def is_goalkeeper(obs, player_num):
    if obs[0]['left_team_roles'][player_num] == 0: # i.e. if goalkeeper
        return True
    return False

def is_opp_goalkeeper(obs, player_num):
    if obs[0]['right_team_roles'][player_num] == 0: # i.e. if goalkeeper
        return True
    return False


def goalkeeper_has_ball(obs):
    ball_owned_player = obs[0]['ball_owned_player'] 
    return is_goalkeeper(obs, ball_owned_player)

GOAL_D_THRESH = 0.5
DEFENDERS_THRESH = 0.07
WINGS_THRESH = 0.5
WINGS_DIFF_WITH_FRONTMOST_PLAYER_THRESH = 0.05
MIDFIELDS_RETREATING_THRESH = DEFENDERS_THRESH*1.5

def close_to_goal(obs, thresh=GOAL_D_THRESH):
    ball_owned_player = obs[0]['ball_owned_player'] 
    x, y = obs[0]['left_team'][ball_owned_player]
    dist_to_goal = dist(x, y, 1, 0)
    # angle_to_goal = angle(x, y, 1, 0)
    # dirn_to_goal = angle_to_direction(angle_to_goal)
    if dist_to_goal <= thresh:# and dirn_to_goal in ['right', 'top-right', 'bottom-right']:
        return True
    return False

def almost_even_closer_to_goal(obs):
    return close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH*1.4)

def even_closer_to_goal(obs):
    return close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH)

def even_more_closer_to_goal(obs):
    return close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH/2)

def close_to_goal_in_x_axis(obs, thresh=GOAL_D_THRESH):
    ball_owned_player = obs[0]['ball_owned_player'] 
    x, y = obs[0]['left_team'][ball_owned_player]
    if x >= 1-thresh:# and dirn_to_goal in ['right', 'top-right', 'bottom-right']:
        return True
    return False

def execute_actions(env, obs, obs_prev, done, actions, ct):
    for action in actions:
        if done:
            reward = [0]*N_PLAYERS_LEFT
            info = None
            break
        obs_prev = obs
        obs, reward, done, info = env.step(action)
        ct += 1
    return obs, obs_prev, reward, done, info, ct

def take_ball_old(env, obs, obs_prev, done, ct):

    x_ball, y_ball, z_ball = obs[0]['ball']
    x_ball_prev, y_ball_prev, z_ball_prev = obs_prev[0]['ball']
    dist_from_ball_to_own_goal = dist(x_ball, y_ball, -1, 0)
    if dist_from_ball_to_own_goal <= MIDFIELDS_RETREATING_THRESH*1.5:
        chosen_metric = 'closest' # if opponents close to goal, closest defender should rush opponent (not one with SID)
    else:
        chosen_metric = 'SID' # otherwise, player with shortest intercept distance (SID) rushes
        
    interceptor_num, isGoalKeeper = identify_best_intercepter(obs, obs_prev, chosen_metric)
    #if DEBUG: print('interceptor num is : ', interceptor_num, ' and isGoalkeeper? ', isGoalKeeper)
    x_int, y_int = obs[0]['left_team'][interceptor_num]
    #if DEBUG: print('attack: ', x_int, y_int, x_ball, y_ball)
    dist_to_ball = dist(x_ball, y_ball, x_int, y_int)

    if dist_to_ball > 0.6:
        if DEBUG: print('anomaly in attack: ', x_int, y_int, x_ball, y_ball)
    
    sprint_action = 13
    if dist_to_ball > TAKE_BALL_THRESH:
        run_angle = angle(x_ball_prev, y_ball_prev, x_ball, y_ball)
        x_ball_future, y_ball_future = ball_pos_in_future(obs, run_angle, dist_to_ball)
        actions = track_position_in_take_ball(obs, x_ball_future, y_ball_future, interceptor_num)
    else:
        actions = track_position_in_take_ball(obs, x_ball, y_ball, interceptor_num, extra_action=sprint_action)

    # if isGoalKeeper:
    #     actions += track_position_in_take_ball(obs, -1+DEFENDERS_THRESH/2, 0, interceptor_num, extra_action=sprint_action)
        
    return execute_actions(env, obs, obs_prev, done, actions, ct)

def take_ball(env, obs, obs_prev, done, ct):
    # just run inbuilt ai
    return do_game_engine(env, obs, obs_prev, done, ct)

def do_nothing(env, obs, obs_prev, done, ct):
    # just stand
    return execute_actions(env, obs, obs_prev, done, [[14]*N_PLAYERS_LEFT], ct)

def do_game_engine(env, obs, obs_prev, done, ct):
    # just run inbuilt ai
    return execute_actions(env, obs, obs_prev, done, [[19]*N_PLAYERS_LEFT], ct)


DIST_TO_BALL_THRESH = 0.1
DIST_TO_OPP_THRESH = 0.2
SHORT_LONG_THRESH = 0.4
SHORT_MIN_THRESH = SHORT_LONG_THRESH/2.0
HIGH_THRESH = 1
TAKE_BALL_THRESH = DIST_TO_BALL_THRESH*10

EVEN_CLOSER_TO_GOAL_THRESH = GOAL_D_THRESH/2

DIST_TO_INTERCEPT_THRESH = 0.3

N_PLAYERS_LEFT = 11
N_PLAYERS_RIGHT = 0
BASE_ACTION = 14

# New: 
SHUFFLE_IN_IDENTIFY_FREE_CHOICE = True
RELAXED_DIST_TO_OPP_THRESH = DIST_TO_OPP_THRESH/2
RELAXED_SHORT_LONG_THRESH = SHORT_LONG_THRESH
RELAXED_SHORT_MIN_THRESH = SHORT_MIN_THRESH/4.0
RELAXED_HIGH_THRESH = HIGH_THRESH/2.0

def pass_ball(env, obs, obs_prev, done, ct):
    actions, reciever_num, kick_angle = pass_to_free_player(obs)
    obs, obs_prev, reward, done, info, ct = execute_actions(env, obs, obs_prev, done, actions, ct)
    return obs, obs_prev, reward, done, info, reciever_num, kick_angle, ct

def pass_ball_to_choice(env, obs, obs_prev, done, ct, choice_num):
    actions, reciever_num, kick_angle = pass_to_free_choice(obs, choice_num)
    obs, obs_prev, reward, done, info, ct = execute_actions(env, obs, obs_prev, done, actions, ct)
    return obs, obs_prev, reward, done, info, reciever_num, kick_angle, ct

def receive_ball(env, obs, obs_prev, done, reciever_num, kick_angle, ct):
    
    x_ball, y_ball, z_ball = obs[0]['ball']
    x_rec, y_rec = obs[0]['left_team'][reciever_num]
    dist_to_ball = dist(x_ball, y_ball, x_rec, y_rec)
    if dist_to_ball > DIST_TO_BALL_THRESH:
        x_ball_future, y_ball_future = ball_pos_in_future(obs, kick_angle, dist_to_ball)
        actions = track_position(obs, x_ball_future, y_ball_future, reciever_num)
    else:
        x_ball, y_ball, z_ball = obs[0]['ball']
        actions = track_position(obs, x_ball, y_ball, reciever_num)

    return execute_actions(env, obs, obs_prev, done, actions, ct)

def move_to_goal(env, obs, obs_prev, done, run_dirn, ct):
    actions = dribble_towards_goal(obs, run_dirn)
    return execute_actions(env, obs, obs_prev, done, actions, ct)

def move_to_goal_with_wings(env, obs, obs_prev, done, ct):
    run_dirn, free_forward_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = False) 
    if not free_forward_space_exists:
        # print('WARNING: **No free space in forward directions (but possible in some direction.)**')
        run_dirn, free_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = False, choose_from_all_directions = True)
        if not free_space_exists:
            # print('WARNING: **Sorry, no free space in any direction, just running right.**')
            run_dirn = 'right'
    
    actions = dribble_towards_goal_with_wings(obs, run_dirn)
    return execute_actions(env, obs, obs_prev, done, actions, ct)

def charge_goal(env, obs, obs_prev, done, ct):
    ball_owned_player = obs[0]['ball_owned_player'] 
    sprint_action = 13
    actions = track_position(obs, 1, 0, ball_owned_player, extra_action=sprint_action) # sprint-to-goal basically
    return execute_actions(env, obs, obs_prev, done, actions, ct)

def just_shoot(env, obs, obs_prev, done, ct):
    ball_owned_player = obs[0]['ball_owned_player'] 
    shoot_action = 12
    actions = [get_list_of_actions(obs, shoot_action, ball_owned_player)]  
    return execute_actions(env, obs, obs_prev, done, actions, ct)

def pass_to_a_player(env, obs, obs_prev, done, ct):
    if DEBUG: print(ct, ' ------------------------------------pass------------------------------------')
    obs, obs_prev, reward, done, info, reciever_num, kick_angle, ct = pass_ball(env, obs, obs_prev, done, ct)
    while not done and no_one_has_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------receive------------------------------------')
        if DEBUG: print(obs[0]['score'])
        obs, obs_prev, reward, done, info, ct = receive_ball(env, obs, obs_prev, done, reciever_num, kick_angle, ct)

    return obs, obs_prev, reward, done, info, ct


def defend(env, obs, obs_prev, done, ct):
    while not done and not have_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------take------------------------------------')
        obs, obs_prev, reward, done, info, ct = take_ball(env, obs, obs_prev, done, ct)
    return obs, obs_prev, reward, done, info, ct

def attack(env, obs, obs_prev, done, ct):
    bool_val = False
    if close_to_goal_in_x_axis(obs):
        # bool_val = True
        obs, obs_prev, reward, done, info, ct = charge_goal(env, obs, obs_prev, done, ct)
        
    if close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH):
        if DEBUG: print(ct, ' ------------------------------------shoot------------------------------------')
        obs, obs_prev, reward, done, info, ct = just_shoot(env, obs, obs_prev, done, ct)
        return obs, obs_prev, reward, done, info, ct
    
    run_dirn_if_free_space, free_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = bool_val) 
    if free_space_exists and not goalkeeper_has_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------move------------------------------------')
        # obs, obs_prev, reward, done, info = move_to_goal(env, obs, obs_prev, run_dirn_if_free_space)
        obs, obs_prev, reward, done, info, ct = move_to_goal_with_wings(env, obs, obs_prev, done, ct)
    else:
        obs, obs_prev, reward, done, info, ct = pass_to_a_player(env, obs, obs_prev, done, ct)
    
    return obs, obs_prev, reward, done, info, ct

def attack_new(env, obs, obs_prev, done, ct):
    bool_val = False
    if close_to_goal_in_x_axis(obs):
        # bool_val = True
        obs, obs_prev, reward, done, info, ct = charge_goal(env, obs, obs_prev, done, ct)
        
        if close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH):
            if DEBUG: print(ct, ' ------------------------------------shoot------------------------------------')
            obs, obs_prev, reward, done, info, ct = just_shoot(env, obs, obs_prev, done, ct)
        return obs, obs_prev, reward, done, info, ct
    
    run_dirn_if_free_space, free_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = bool_val) 
    if free_space_exists and not goalkeeper_has_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------move------------------------------------')
        # obs, obs_prev, reward, done, info = move_to_goal(env, obs, obs_prev, run_dirn_if_free_space)
        obs, obs_prev, reward, done, info, ct = move_to_goal_with_wings(env, obs, obs_prev, done, ct)
    else:
        obs, obs_prev, reward, done, info, ct = pass_to_a_player(env, obs, obs_prev, done, ct)
    
    return obs, obs_prev, reward, done, info, ct

def maintain_ball_possession(env, obs, obs_prev, done, ct):
    run_dirn_if_free_space, free_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = False) 
    if free_space_exists and not goalkeeper_has_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------move------------------------------------')
        obs, obs_prev, reward, done, info, ct = move_to_goal_with_wings(env, obs, obs_prev, done, ct)
    else:
        obs, obs_prev, reward, done, info, ct = pass_to_a_player(env, obs, obs_prev, done, ct)
    
    return obs, obs_prev, reward, done, info, ct

def goal_scored(obs, obs_prev):
    prev_score = obs_prev[0]['score'][0]
    new_score = obs[0]['score'][0]
    # print(prev_score_diff, score_diff)
    if new_score - prev_score == 1: 
        return True
    else:
        return False

def opp_goal_scored(obs, obs_prev):
    prev_score_diff = obs_prev[0]['score'][0] - obs_prev[0]['score'][1]
    score_diff = obs[0]['score'][0] - obs[0]['score'][1]
    if score_diff - prev_score_diff == -1: 
        return True
    else:
        return False

def ball_in_opp_half(obs):
    x_ball, y_ball, z_ball = obs[0]['ball']
    if x_ball >= 0:
        return True
    return False

def ball_in_opp_quarter(obs):
    x_ball, y_ball, z_ball = obs[0]['ball']
    if x_ball >= 0.5:
        return True
    return False

def pass_to_choice(env, obs, obs_prev, done, ct, choice_num):
    if DEBUG: print(ct, ' ------------------------------------pass------------------------------------')
    obs, obs_prev, reward, done, info, reciever_num, kick_angle, ct = pass_ball_to_choice(env, obs, obs_prev, done, ct, choice_num)
    while not done and no_one_has_ball(obs):
        if DEBUG: print(ct, ' ------------------------------------receive------------------------------------')
        if DEBUG: print(obs[0]['score'])
        obs, obs_prev, reward, done, info, ct = receive_ball(env, obs, obs_prev, done, reciever_num, kick_angle, ct)

    return obs, obs_prev, reward, done, info, ct

def common_sense_baseline(env, obs, obs_prev, done, ct):
    if not have_ball(obs):
        obs, obs_prev, reward, done, info, ct = defend(env, obs, obs_prev, done, ct)
    elif have_ball(obs):
        obs, obs_prev, reward, done, info, ct = attack_new(env, obs, obs_prev, done, ct)
    return obs, obs_prev, reward, done, info, ct

