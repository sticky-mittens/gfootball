import gfootball.env as football_env
from controllers.util import DQN, ReplayMemory, Transition
import torch
from torch import optim, nn
import math 
import random
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
from pass_to_free_0 import precondition, postcondition, short_pass_to_free_player, long_pass_to_free_player, high_pass_to_free_player


# gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setup env
env = football_env.create_environment(
    env_name='11_vs_11_stochastic',
    representation='raw',
    rewards='checkpoints,scoring',
    write_full_episode_dumps=False,
    write_video=False,
    logdir='logs_new_v2/',
    render=True,
    other_config_options={'physics_steps_per_frame': 4},
    )
#env.reset()

# Get number of actions from gym action space, setup nets + optimizer + replay memory
n_actions = 3 # short, long, high
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

# training loop
NUM_EPISODES = 2
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
