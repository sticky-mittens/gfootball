import gfootball.env as football_env
from controllers.util import DQN, ReplayMemory, Transition
import torch
from torch import optim, nn
import math 
import random
from itertools import count
import matplotlib.pyplot as plt
import numpy as np


# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup env
env = football_env.create_environment(
    env_name='11_vs_11_stochastic',
    representation='simple115v2',
    rewards='checkpoints,scoring',
    write_full_episode_dumps=True,
    write_video=True,
    logdir='logs_new_v2/',
    render=True)
#env.reset()

# Get number of actions from gym action space, setup nets + optimizer + replay memory
n_actions = 11 # for each player that agent can pass to (including itself)
print('No of actions are: ', n_actions)

policy_net = DQN(115, n_actions).to(device)
target_net = DQN(115, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

# setup epsilon-greedy policy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

steps_done = 0
def policy(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # Second column (aka [1]) of above is index of where max element was found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1) # view is like reshape but safer for tensors
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# One training iteration
BATCH_SIZE = 128
GAMMA = 0.999

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. 
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch) # print(action_batch.shape, policy_net(state_batch).shape) # = torch.Size([128, 1]),  torch.Size([128, 19])

    # Compute V(s_{t+1}) = \max_{a} Q(s_{t+1}, a) for all next states.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
    # [Since by defn, V(final state) = 0] This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final. 
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def pass_controller(reciever_number, state, reward, done, info):
    # print(state.shape) # = torch.Size([1, 115])
    reciever_orientation = state[:, 22+2*reciever_number: 22+2*reciever_number+2] # torch.Size([1, 2])
    agent_number = torch.argmax(state[:, -7-11:-7])
    angle_radians = torch.atan2(reciever_orientation[:, 1], reciever_orientation[:, 0]).item()
    angle_degrees = angle_radians * 180.0 / math.pi
    angle_degrees = max(-180, min(180, angle_degrees))

    goal_location = [1, 0]
    agent_position = state[:, 2*agent_number: 2*agent_number+2]
    dist_to_goal = torch.sqrt( (goal_location[1] - agent_position[:, 1])**2 + (goal_location[0] - agent_position[:, 0])**2 ).item()

    SHOOT_THRESHOLD = 0.5
    SHORT_PASS_THRESHOLD = 0.5

    reciever_position = state[:, 2*reciever_number: 2*reciever_number+2]
    distance = torch.sqrt( (reciever_position[:, 1] - agent_position[:, 1])**2 + (reciever_position[:, 0] - agent_position[:, 0])**2 ).item()
    
    action_map = {'left': 1, 'top-left': 2, 'top': 3, 'top-right': 4, 'right': 5, 'bottom-right':6, 'bottom': 7, 'bottom-left': 8,
                    'long-pass':9, 'high-pass?':10, 'short-pass':11}
    actions = []
    if agent_number == reciever_number:
        actions.append(0)#action_map['right'])
    elif dist_to_goal <= SHOOT_THRESHOLD:
        actions.append(12)
    else:
        if angle_degrees > 0 and angle_degrees < 90:
            direction = 'top-right'
        elif angle_degrees > 90 and angle_degrees < 180:
            direction = 'top-left'
        elif angle_degrees < 0 and angle_degrees > -90:
            direction = 'bottom-right'
        elif angle_degrees < -90 and angle_degrees > -180:
            direction = 'bottom-left'
        elif angle_degrees == 90:
            direction = 'top'
        elif angle_degrees == -90:
            direction = 'bottom'
        elif angle_degrees == 0:
            direction = 'left'
        elif angle_degrees == 180 or angle_degrees == -180:
            direction = 'right'
        #print(angle_degrees)
        actions.append(action_map[direction])

        if distance <= SHORT_PASS_THRESHOLD:
            pass_type = 'short-pass'
        else:
            pass_type = 'long-pass'

        actions.append(action_map[pass_type])

    for action in actions:
        if not done:
            state, reward, done, info = env.step(action)

    return state, reward, done, info

def is_opponent_posession(state):
    one_hot_posession = state[:, -3-11-7:-11-7] # noone, left (us), right (opponent)
    if torch.argmax(one_hot_posession).item() == 2:
        return True
    else:
        return False

# training loop
NUM_EPISODES = 50
TARGET_UPDATE_C = 5
LOG_INTERVAL = 500
episode_durations = []
for episode in range(NUM_EPISODES):
    env.reset()
    state, reward, done, info = env.step(0)
    state = torch.tensor([state], device=device, dtype=torch.float)
    for t in count():
        # Select and perform an action
        player_to_pass_to = policy(state)
        # internally use pass controller to complete pass and get outputs
        next_state, reward, done, info = pass_controller(player_to_pass_to, state, reward, done, info)
        next_state = torch.tensor([next_state], device=device, dtype=torch.float)

        # # check if ball is in not in our possession, and if so, change reward to -1 (or other way with +1)
        # if is_opponent_posession(next_state):
        #     reward -= 1
        
        if t%LOG_INTERVAL == 0 or reward not in [-1.0, 0.0]:
            print('Episode {0} | Time {1} | Reward {2}'.format(episode, t, reward))
        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, player_to_pass_to, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        #optimize_model_simple()
        if done:
            episode_durations.append(t + 1)
            print('Episode {0} took {1} time to complete.'.format(episode, t))
            break

    # Update the target network every C episodes, copying all weights and biases in policy network
    if episode % TARGET_UPDATE_C == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
plt.ioff()
plt.show()
