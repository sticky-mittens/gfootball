import gfootball.env as football_env
import random
import numpy as np
import math
from controllers.teleports import maintain_ball_possession, no_one_has_ball, close_to_goal_in_x_axis, have_ball, pass_to_a_player, take_ball, close_to_goal, identify_free_space, move_to_goal_with_wings, pass_ball, receive_ball, charge_goal, just_shoot, EVEN_CLOSER_TO_GOAL_THRESH, goalkeeper_has_ball

N_PLAYERS_LEFT = 11
N_PLAYERS_RIGHT = 0

env = football_env.create_environment(
    env_name='11_vs_11_easy_stochastic',#'academy_3_vs_1_with_keeper',#'11_vs_11_easy_stochastic',#
    representation='raw',#'simple115v2',
    rewards='checkpoints,scoring',
    write_full_episode_dumps=True,
    write_video=True,
    logdir='logs/',
    number_of_left_players_agent_controls=N_PLAYERS_LEFT,
    number_of_right_players_agent_controls=N_PLAYERS_RIGHT,
    render=True,
    other_config_options={'physics_steps_per_frame': 3, 
                            'video_quality_level': 1,
                            'action_set': 'v2',
                            'display_game_stats': False
                        },
    )


if __name__ == "__main__":
    while True:
        obs = env.reset()
        obs_prev = obs
        done = False
        ct = 0
        while not done:
            if not have_ball(obs):
                while not done and not have_ball(obs):
                    print(ct, ' ------------------------------------attack------------------------------------')
                    obs, obs_prev, reward, done, info, ct = take_ball(env, obs, obs_prev, done, ct)
            elif have_ball(obs):
                bool_val = False
                if close_to_goal_in_x_axis(obs):
                    # bool_val = True
                    obs, obs_prev, reward, done, info, ct = charge_goal(env, obs, obs_prev, done, ct)
                    
                    if close_to_goal(obs, thresh=EVEN_CLOSER_TO_GOAL_THRESH):
                        print(ct, ' ------------------------------------shoot------------------------------------')
                        obs, obs_prev, reward, done, info, ct = just_shoot(env, obs, obs_prev, done, ct)
                    continue
                
                run_dirn_if_free_space, free_space_exists = identify_free_space(obs, ct, constrained_to_goal_directions = bool_val) 
                if free_space_exists and not goalkeeper_has_ball(obs):
                    print(ct, ' ------------------------------------move------------------------------------')
                    obs, obs_prev, reward, done, info, ct = move_to_goal_with_wings(env, obs, obs_prev, done, ct)
                else:
                    obs, obs_prev, reward, done, info, ct = pass_to_a_player(env, obs, obs_prev, done, ct)