from audioop import avg
from collections import defaultdict
from email.policy import default
import sys
sys.path.append("../")

import torch
import numpy as np
import math
from torch import nn, optim
import random
import copy
import sys
import os
from torch.autograd import Variable
import torch.nn.functional as F
from matplotlib import pyplot as plt
from controllers import teleports
from scripts import utils_orig as utils
import pickle
from collections import defaultdict
import heapq
import time

class ReplayMemory:
    def __init__(self, n_episodes):
        self.n_epsiodes = n_episodes
        self.memory = defaultdict(list)
        self.ep_scorediff_tuples_heap = [] # contains tuples like (ep_num, score_diff)

    def push(self, ep_num, transition):
        self.memory[ep_num].append(transition)
        if len(self.memory.keys()) > self.n_epsiodes:
            least_score_diff, ep_to_delete = heapq.heappop(self.ep_scorediff_tuples_heap) 
            print(f'== current size of memory is eps {len(self.memory.keys())} > {self.n_epsiodes} and we are deleting ep {ep_to_delete}')
            del self.memory[ep_to_delete]

    def sample(self, batch_size):
        all_available_memories = []
        for ep_num in self.memory.keys():
            all_available_memories.extend(self.memory[ep_num])
        return random.sample(all_available_memories, batch_size)

    def __len__(self):
        tot_len = 0
        for ep_num in self.memory.keys():
            tot_len += len(self.memory[ep_num])
        return tot_len

class policyNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5):
        self.no_of_layers = 2
        self.no_of_hidden_units = 100 
        super(policyNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())

        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))
        self.lin_trans.append(torch.nn.Sigmoid())
        self.lin_trans.append(torch.nn.Softmax())

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class criticNet(nn.Module):
    def __init__(self, no_of_inputs = 10, no_of_outputs = 5, no_of_layers = 5, no_of_hidden_units = 500):
        self.no_of_layers = no_of_layers # OG: 5
        self.no_of_hidden_units = no_of_hidden_units # OG: 500

        super(criticNet, self).__init__()

        self.lin_trans = []

        self.lin_trans.append(nn.Linear(no_of_inputs, self.no_of_hidden_units))
        self.lin_trans.append(torch.nn.ReLU())
        for layer_index in range(self.no_of_layers-2):
            self.lin_trans.append(nn.Linear(self.no_of_hidden_units, self.no_of_hidden_units))
            self.lin_trans.append(torch.nn.ReLU())
        self.lin_trans.append(nn.Linear(self.no_of_hidden_units, no_of_outputs))

        self.model = torch.nn.Sequential(*self.lin_trans)

    def forward(self, x):
        y = self.model(x)
        return y

class Hierarchical_Policies():
    ''' 1. Defines policies for each level
    '''
    def __init__(self, level_num, OT_name, folder_name = 'saved_models', use_demos = False, gamma = 0.9, learn_freq = 1, critic_lr = 0.001, mem_size = 20, n_layers = 5, n_hidden_units = 500): #, use_target = False, target_update = 1):
        self.level_num = level_num
        self.OT_name = OT_name
        self.folder_name = folder_name
        # self.use_target = use_target
        self.learn_freq = learn_freq
        # print(f'==> use target? {self.use_target}')

        self.plot_losses = True
        self.losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'==>using device {self.device}')
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
        self.Tensor = self.FloatTensor

        self.n_inputs = 139

        if level_num == 1:
            self.n_actions = 2
        elif level_num == 2:
            self.n_actions = 3
        elif level_num == 3:
            self.n_actions = 2
        elif level_num == 4:
            self.n_actions = 5

        self.EPS_START = 0.9  # e-greedy threshold start value
        self.EPS_END = 0.001 # e-greedy threshold end value
        self.EPS_DECAY = 20  # OG: 200, then 25 # e-greedy threshold decay
        self.GAMMA = gamma  # Q-learning discount factor # gamma needs to be small (so it doesnt score at end.) # print rewards vs time in one ep to see when reward is given
        self.critic_LR = critic_lr  # NN optimizer learning rate

        self.POST_CDN_TIMESTEP_LIMIT = 200

        self.critic = criticNet(no_of_inputs = self.n_inputs, no_of_outputs = self.n_actions, no_of_layers = n_layers, no_of_hidden_units = n_hidden_units).to(self.device)
        print(f'==>critic has {n_layers} layers and {n_hidden_units} hidden units.')

        # self.critic_target = criticNet(self.n_inputs, self.n_actions).to(self.device)
        # self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.critic_LR)

        self.memory = ReplayMemory(n_episodes = mem_size) # OG: 10000 # keeps approx 20-25 episodes each with around 500/400 learning steps
        # self.tau = 0.005 # for critic target update
        # self.TARGET_UPDATE = target_update

        self.use_demos = use_demos
        if self.use_demos:
            print(f'==> Using some demos loaded from ./demos.pkl .')
            with open(f'demos.pkl', 'rb') as file:
                self.demos = pickle.load(file)

    def select_action_epsilon_greedy(self, state, critic, ep_num):
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * ep_num / self.EPS_DECAY)
        if sample > self.eps_threshold:
            with torch.no_grad():
            # t.max(1) will return (largest column value of each row, its index).
            # we return this index of max element, reshaped/viewed as (1,1)
                return critic(state).max(1)[1].view(1, 1)
        else:
            return self.LongTensor([[random.randrange(self.n_actions)]])

    def select_action(self, state, critic):
        with torch.no_grad():
        # t.max(1) will return (largest column value of each row, its index).
        # we return this index of max element, reshaped/viewed as (1,1)
            return critic(state).max(1)[1].view(1, 1)

    def old_learn(self):
        self.BATCH_SIZE = math.ceil(0.2 * len(self.memory))

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        if self.use_demos:
            self.DEMOS_BATCH_SIZE = math.ceil(0.25 * self.BATCH_SIZE)
            demo_transitions = self.demos.sample(self.DEMOS_BATCH_SIZE)
            transitions += demo_transitions
            self.BATCH_SIZE += self.DEMOS_BATCH_SIZE
            

        # convert batch of transitions (which is a list of tuples) to transition of batches 
        ## (*list) = unpacks list = removes []; E.G. (*[tup1, tup2, ...]) = (tup1, tup2, ...)
        ## zip((a1, a2, a3, a4), (b1, b2, b3, b4), ...) = [(a1, b1, ...), (a2, b2, ...), (a3, b3, ...), (a4, b4, ...)]
        batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions) 
        ## Next, we concatenate tuple (a1, b1, ...) into tensor [a1 b1 ...]
        batch_state = Variable(torch.cat(batch_state))
        batch_action = Variable(torch.cat(batch_action))
        batch_reward = Variable(torch.cat(batch_reward))
        batch_next_state = Variable(torch.cat(batch_next_state)) 

        # print(batch_state.shape, batch_action.shape, batch_next_state.shape, batch_reward.shape) # (B, 139), (B, 1), (B, 139), (B, 1)
        # print(self.critic(batch_state).shape) # (B, self.n_actions)

        # Estimate Q(s, a), max_a' Q(s', a') and Expected value
        ## gets value of critic(batch_state)'s rows at index given by batch_action's elements # (B, 1)
        Q_s_a = self.critic(batch_state).gather(1, batch_action) 
        ## *detaches* and gets max value of critic(batch_state)'s rows # (B, 1)
        # max_Q_s_next_a_next = self.critic_target(batch_next_state).detach().max(1)[0].view(self.BATCH_SIZE, 1)
        max_Q_s_next_a_next = self.critic(batch_next_state).detach().max(1)[0].view(self.BATCH_SIZE, 1)
        expected_Q = batch_reward + (self.GAMMA * max_Q_s_next_a_next)
        # print(f'for comparison of max: E[Q] {torch.max(expected_Q)}, R {torch.max(batch_reward)}, discounted {torch.max(self.GAMMA * max_Q_s_next_a_next)}')
        # print(f'for comparison of min: E[Q] {torch.min(expected_Q)}, R {torch.min(batch_reward)}, discounted {torch.min(self.GAMMA * max_Q_s_next_a_next)}\n')

        # loss is measured from error between current and newly expected Q values
        # print(Q_s_a.shape, max_Q_s_next_a_next.shape, batch_reward.shape, expected_Q.shape) # all (B,1) 
        loss = F.smooth_l1_loss(Q_s_a, expected_Q)
        if self.plot_losses: self.losses.append(loss.item())

        # backpropagation of loss to NN
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        self.BATCH_SIZE = math.ceil(0.4 * len(self.memory))

        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.BATCH_SIZE)
        if self.use_demos:
            self.DEMOS_BATCH_SIZE = math.ceil(0.25 * self.BATCH_SIZE)
            demo_transitions = self.demos.sample(self.DEMOS_BATCH_SIZE)
            transitions += demo_transitions
            self.BATCH_SIZE += self.DEMOS_BATCH_SIZE
        
        # print(batch_state.shape, batch_action.shape, batch_next_state.shape, batch_reward.shape) # (B, 139), (B, 1), (B, 139), (B, 1)
        # print(self.critic(batch_state).shape) # (B, self.n_actions)

        critic_loss_list = []
        for index in range(self.BATCH_SIZE):
            s, a, next_s, r, terminal = transitions[index]
            # print('s, a, r, next_s ', s, a, r, next_s)
            prediction = self.critic(s) # (1, n_actions)
            
            if terminal:
                target_val = r
            else:
                next_val, next_action = self.critic(next_s).max(1)
                # if self.use_target:
                #     next_val, next_action = self.critic_target(next_s).max(1)
                # else:
                #     next_val, next_action = self.critic(next_s).max(1)
                # # print('next_val, next_action ', next_val, next_action)
                target_val = r * 1.0 + self.GAMMA * next_val.clone().detach().cpu().item() * 1.0
            # print(f'==== learning step {self.t}, prediction {torch.max(prediction.clone().detach())}, reward {r}, target_val {target_val}')

            list_target = [0]*self.n_actions
            list_target[a] = target_val
            torch_target = self.FloatTensor([list_target]).to(self.device) # (1, n_actions)

            loss = self.criterion(prediction, torch_target).to(self.device)
            # print('loss ', loss) # is a tensor([float]) # single number i.e.
            critic_loss_list.append(loss)

        # backpropagation of loss to NN
        self.critic_optimizer.zero_grad()
        critic_loss = torch.stack(critic_loss_list, dim=0).sum()
        if self.plot_losses: self.losses.append(critic_loss.clone().item())
        critic_loss.backward()
        self.critic_optimizer.step()

    def goal_scored(self, obs):
        new_score = obs[0]['score'][0]
        # print(self.our_score, new_score)
        if new_score - self.our_score == 1:
            self.our_score = new_score
            return True
        else:
            self.our_score = new_score
            return False


    def update_policy(self, env, obs, obs_prev, done, shaped_reward_function, learnt_options_set, ep_num, debug):
        learnt_options_set[self.OT_name] = {'pre': utils.all_OTs[self.level_num][self.OT_name]['pre'], 
                                            'post': utils.all_OTs[self.level_num][self.OT_name]['post'], 
                                            'policy': self.critic, 
                                            'deterministic': False
                                            }

        obs_wrapped = utils.obs_wrapper(obs) # (1, 139) 2D list
        obs_pre_teleport = obs.copy()
        obs_wrapped_pre_teleport = obs_wrapped.copy()
        obs_wrapped_tensor = self.FloatTensor(obs_wrapped).to(self.device)

        action_OT_number = self.select_action_epsilon_greedy(obs_wrapped_tensor, self.critic, ep_num)
        action_OT_number = action_OT_number.cpu().item() 
        action_OT_name = utils.convert_number_to_name(self.level_num, action_OT_number)
        if debug: print(f'{self.level_num}) {self.OT_name} while learning, chose {action_OT_name}')

        start_time = self.ct
        goal_scored = False
        while True:
            if utils.all_OTs[self.level_num+1][action_OT_name]['pre'](obs): 
                obs, obs_prev, reward, done, info, self.ct = utils.all_OTs[self.level_num+1][action_OT_name]['OT'](env, obs, obs_prev, done, self.ct)
                if self.goal_scored(obs):
                    print('goal_identified')
                    goal_scored = True
                    break
                
                if done or utils.all_OTs[self.level_num+1][action_OT_name]['post'](obs) or (self.ct - start_time > self.POST_CDN_TIMESTEP_LIMIT):
                    break
            else:
                break
        
        shaped_reward = shaped_reward_function(obs, obs_prev, action_OT_name, reward)

        if goal_scored:
            shaped_reward += 1 # less maybe!

        # # below for old_learn
        # self.memory.push(ep_num, (self.FloatTensor(obs_wrapped_pre_teleport).to(self.device),
        #         self.LongTensor([[action_OT_number]]).to(self.device), 
        #         self.FloatTensor(utils.obs_wrapper(obs)).to(self.device),
        #         self.FloatTensor([[shaped_reward]]).to(self.device) 
        #         ))

        # # below 3 lines for new_learn
        self.memory.push(ep_num, (self.FloatTensor(obs_wrapped_pre_teleport).to(self.device), # (1, 139)
                int(action_OT_number), # self.LongTensor([[action_OT_number]]).to(self.device), # integer
                self.FloatTensor(utils.obs_wrapper(obs)).to(self.device), # (1, 139)
                shaped_reward, # self.FloatTensor([[shaped_reward]]).to(self.device)  # float
                True if done else False
                ))
    
        # self.learn() 

        return obs, obs_prev, shaped_reward, done, info

    def LearnGuidedPolicy(self, env, shaped_reward_function, learnt_options_set, ep_num, debug = True):
        # if (ep_num+1) % self.TARGET_UPDATE == 0:
        #     for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #         target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # # OLd update method below
            # self.critic_target.load_state_dict(self.critic.state_dict())

        obs = env.reset()
        obs_prev = obs
        start_time_ep = time.time()

        self.our_score = 0 # to check if goal scored
        self.criterion = nn.SmoothL1Loss()

        execution_stack = []
        self.t = 0
        self.ct = 0

        execution_stack.append((1, 'win_game'))
        self.shaped_rewards_list = []

        done = False
        while not done:
            top_option_level, top_option_name = execution_stack[-1]

            if top_option_name == self.OT_name:
                if debug: print(f'{self.level_num}) {top_option_name} being learnt\n')
                obs, obs_prev, shaped_reward, done, info = self.update_policy(env, obs, obs_prev, done, shaped_reward_function, learnt_options_set, ep_num, debug)

                # printing info
                self.shaped_rewards_list.append(shaped_reward)
                # print(f'step: {self.t}, reward {shaped_reward} (env step for info: {self.ct})')

                if utils.all_OTs[self.level_num][self.OT_name]['post'](obs):
                    if debug: print(f'{self.level_num}) {self.OT_name} learnt until termination and popped')
                    execution_stack.pop()
                
            else:
                # get top policy and wrapped observations 
                '''
                2) when learning attack on level 2, win_game's determinstic policy is obtained

                3a) when learning pass_to_a_player on level 3, win_game's determinstic policy is obtained
                                                           
                3b)                                           attack's NN policy is obtained
                '''
                top_options_policy = learnt_options_set[top_option_name]['policy']
                
                obs_wrapped = utils.obs_wrapper(obs)
                obs_wrapped_tensor = self.FloatTensor(obs_wrapped).to(self.device)
                
                # get name of action OT 
                '''
                2) when learning attack on level 2, win_game chooses either attack or defend

                3a) when learning pass_to_a_player on level 3, win_game chooses either attack or defend
                                                           
                3b)                                            attack chooses from charge_goal, just_shoot, move_to_goal_with_wings and pass_to_a_player
                '''
                if learnt_options_set[top_option_name]['deterministic']:
                    action_OT_name = top_options_policy(obs)
                    if debug: print(f'{self.level_num}) {top_option_name} chose {action_OT_name}')
                else:
                    action_OT_number = self.select_action(obs_wrapped_tensor, top_options_policy)
                    action_OT_number = action_OT_number.cpu().item()
                    action_OT_name = utils.convert_number_to_name(top_option_level, action_OT_number)

                # execute it if isAvailable(e) or else add to exec stack
                '''
                2) when learning attack on level 2, if attack is chosen - it isn't available at level 3 - push to execution stack and learn in next step
                                                    if defend is chosen - defend_ is available in level 3 - execute it (and pop if post condition reached)

                3a) when learning pass_to_a_player on level 3, if attack is chosen - it isn't available at level 4 - push (2, attack) to execution stack 
                                                               if defend is chosen - defend__ is available in level 4 - execute it (and pop if post condition reached)

                3b)                                            if charge_goal is chosen             - it's available in level 4 - execute
                                                               if just_shoot is chosen              - it's available in level 4 - execute
                                                               if move_to_goal_with_wings is chosen - it's available in level 4 - execute
                                                               if pass_to_a_player is chosen        - it isn't available in level 4 - push (3, pass_to_a_player) to execution stack and learn it in next step

                                                        
                '''
                if action_OT_name in utils.all_OTs[self.level_num+1]:

                    if utils.all_OTs[self.level_num+1][action_OT_name]['pre'](obs):
                        if debug: print(f'{self.level_num}) {action_OT_name} executed')
                        obs, obs_prev, _, done, info, self.ct = utils.all_OTs[self.level_num+1][action_OT_name]['OT'](env, obs, obs_prev, done, self.ct)
                    
                    if utils.all_OTs[self.level_num+1][action_OT_name]['post'](obs): # TODO # add time to all post conditions
                        if debug: print(f'{self.level_num}) {action_OT_name} terminated and popped')
                        execution_stack.pop() 
                else:

                    if utils.all_OTs[top_option_level+1][action_OT_name]['pre'](obs):
                        if debug: print(f'{self.level_num}) {action_OT_name} pushed to stack')
                        execution_stack.append((top_option_level+1, action_OT_name))

        sum_reward = sum(self.shaped_rewards_list)
        n_steps = len(self.shaped_rewards_list)
        current_score_diff = obs[0]['score'][0]-obs[0]['score'][1]
        heapq.heappush(self.memory.ep_scorediff_tuples_heap, (current_score_diff, ep_num)) # heapq first compares tuples by first value, then by second value to break tie
        
        if (ep_num+1) % self.learn_freq == 0:
            while self.t < 10:
                self.learn() 
                self.t += 1
            
        print(f'=== ep: {ep_num}, time {time.time()-start_time_ep}, eps {self.eps_threshold}, sum reward: {sum_reward}, score_diff {current_score_diff}, tot learning steps {self.t} (total env steps {self.ct})')


        if self.plot_losses and (ep_num+1):
            plt.figure()
            plt.plot(self.losses)
            plt.ylim([0, 40])
            plt.title(f'ep {ep_num}')

            os.makedirs(f'./{self.folder_name}', exist_ok = True)
            os.makedirs(f'./{self.folder_name}/loss_plots', exist_ok = True)

            plt.savefig(f'./{self.folder_name}/loss_plots/{ep_num+1}.png')
        return sum_reward


    def save_to_disk(self):
        os.makedirs(f'./{self.folder_name}', exist_ok = True)
        torch.save(self.critic, f'./{self.folder_name}/{self.OT_name}.pt')

    def save_to_disk_extra(self, R, ep_num):
        os.makedirs(f'./{self.folder_name}', exist_ok = True)
        torch.save(self.critic, f'./{self.folder_name}/{self.OT_name}_{R}_{ep_num}.pt')
