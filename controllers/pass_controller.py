import gfootball.env as football_env
import random
import numpy as np

env = football_env.create_environment(
    env_name='11_vs_11_stochastic',
    representation='simple115v2',
    rewards='checkpoints,scoring',
    render=True)
env.reset()

N_PLAYERS = 11
done = False 
obs, _, _, _ = env.step(0)
while not done:
    reciever_number = int(random.randrange(N_PLAYERS))
    #reciever_position = obs[2*reciever_number: 2*reciever_number+2]
    reciever_orientation = obs[22+2*reciever_number: 22+2*reciever_number+2]
    agent_number = np.argmax(obs[-7-11:-7])
    #agent_position = obs[2*agent_number: 2*agent_number+2]
    #agent_orientation = obs[22+2*agent_number: 22+2*agent_number+2]
    angle_radians = np.arctan2(reciever_orientation[1], reciever_orientation[0])
    angle_degrees = angle_radians * 180 / np.pi
    #distance = np.sqrt( (reciever_position[1] - agent_position[1])**2 + (reciever_position[0] - agent_position[0])**2 )
    
    action_map = {'left': 1, 'top-left': 2, 'top': 3, 'top-right': 4, 'right': 5, 'bottom-right':6, 'bottom': 7, 'bottom-left': 8}
    if agent_number == reciever_number:
        actions = [0]
    else:
        if angle_degrees > 0 and angle_degrees < 90:
            direction = 'top-right'
        elif angle_degrees > 90 and angle_degrees < 180:
            direction = 'top-left'
        elif angle_degrees < 0 and angle_degrees > -90:
            direction = 'bottom-right'
        elif angle_degrees < -90 and angle_degrees > -180:
            direction = 'bottom-left'
        actions = [action_map[direction], 9]

    for action in actions:
        obs, reward, done, info = env.step(action)


