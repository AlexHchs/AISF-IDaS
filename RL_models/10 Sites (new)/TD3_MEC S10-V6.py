#!/usr/bin/env python
# coding: utf-8

# # Twin-Delayed DDPG

# ## Installing the packages

#
# In[2]:


import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from MECenvirontment_TD3_V2 import EnvironmentTD3
from sklearn import preprocessing


### Step 1: We initialize the Experience Replay memory

# In[3]:


class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0
    print("Storage : ", self.storage)
  def add(self, transition):
    # if the memory has been populated so it placed in the beginning (replace the old) 
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


### Step 2: We build one neural network for the Actor model and one neural network for the Actor target

# In[4]:


class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 84)
    self.layer_2 = nn.Linear(84, 42)
    self.layer_3 = nn.Linear(42, 42)
    
    self.layer_11 = nn.Linear(42, 84)
    self.layer_12 = nn.Linear(84, 42)
    self.layer_13 = nn.Linear(42, action_dim)
    self.layer_21 = nn.Linear(42, 84)
    self.layer_22 = nn.Linear(84, 42)
    self.layer_23 = nn.Linear(42, action_dim)
    self.layer_31 = nn.Linear(42, 84)
    self.layer_32 = nn.Linear(84, 42)
    self.layer_33 = nn.Linear(42, action_dim)
    self.layer_41 = nn.Linear(42, 84)
    self.layer_42 = nn.Linear(84, 42)
    self.layer_43 = nn.Linear(42, action_dim) 
    self.layer_51 = nn.Linear(42, 84)
    self.layer_52 = nn.Linear(84, 42)
    self.layer_53 = nn.Linear(42, action_dim)
    self.layer_61 = nn.Linear(42, 84)
    self.layer_62 = nn.Linear(84, 42)
    self.layer_63 = nn.Linear(42, action_dim)
    self.layer_71 = nn.Linear(42, 84)
    self.layer_72 = nn.Linear(84, 42)
    self.layer_73 = nn.Linear(42, action_dim)
    self.layer_81 = nn.Linear(42, 84)
    self.layer_82 = nn.Linear(84, 42)
    self.layer_83 = nn.Linear(42, action_dim)
    self.layer_91 = nn.Linear(42, 84)
    self.layer_92 = nn.Linear(84, 42)
    self.layer_93 = nn.Linear(42, action_dim)
    
    
  def forward(self, x): #x is the input state
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = F.relu(self.layer_3(x))
    ####################################################################################
    y1 = F.relu(self.layer_11(x))
    y1 = F.relu(self.layer_12(y1))
    # y1 = self.layer_13(y1)
    y1 = F.softmax(self.layer_13(y1),dim=1)
    
    y2 = F.relu(self.layer_21(x))
    y2 = F.relu(self.layer_22(y2))
    # y2 = self.layer_23(y2)
    y2 = F.softmax(self.layer_23(y2),dim=1)
     
    y3 = F.relu(self.layer_31(x))
    y3 = F.relu(self.layer_32(y3))
    # y3 = self.layer_33(y3)
    y3 = F.softmax(self.layer_33(y3),dim=1)
    
    y4 = F.relu(self.layer_41(x))
    y4 = F.relu(self.layer_42(y4))
    # y4 = self.layer_43(y4)
    y4 = F.softmax(self.layer_43(y4),dim=1)
    
    y5 = F.relu(self.layer_51(x))
    y5 = F.relu(self.layer_52(y5))
    # y5 = self.layer_53(y5)
    y5 = F.softmax(self.layer_53(y5),dim=1)
    
    y6 = F.relu(self.layer_61(x))
    y6 = F.relu(self.layer_62(y6))
    # y6 = self.layer_63(y6)
    y6 = F.softmax(self.layer_63(y6),dim=1)
     
    y7 = F.relu(self.layer_71(x))
    y7 = F.relu(self.layer_72(y7))
    # y7 = self.layer_73(y7)
    y7 = F.softmax(self.layer_73(y7),dim=1)
    
    y8 = F.relu(self.layer_81(x))
    y8 = F.relu(self.layer_82(y8))
    # y8 = self.layer_83(y8)
    y8 = F.softmax(self.layer_83(y8),dim=1)
    # print("y1", y1)
    y9 = F.relu(self.layer_91(x))
    y9 = F.relu(self.layer_92(y9))
    # y5 = self.layer_53(y5)
    y9 = F.softmax(self.layer_93(y9),dim=1)
    ####################################################################################

    
    
    return y1,y2,y3,y4,y5,y6,y7,y8,y9
    

   
# ## Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets

# In[5]:


class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim_critic):
    super(Critic, self).__init__()
    self.layer_1 = nn.Linear(state_dim + action_dim_critic, 134)
    self.layer_2 = nn.Linear(134, 67)
    self.layer_3 = nn.Linear(67, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim_critic, 134)
    self.layer_5 = nn.Linear(134, 67)
    self.layer_6 = nn.Linear(67, 1)

  def forward(self, x, u): #x is state, u is action
    xu = torch.cat([x, u], 1) #vertical concantecated 
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1


# ## Steps 4 to 15: Training Process

# In[6]:


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, action_dim_critic):
    self.actor = Actor(state_dim, action_dim).to(device)
    self.actor_target = Actor(state_dim, action_dim).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim_critic).to(device)
    self.critic_target = Critic(state_dim, action_dim_critic).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    xl = self.actor(state)
    
    lst = []
    n = env.ANNum
    for i in xl:
        lst.append(i.cpu().data.numpy().flatten())
    zl = [lst[i:i + n] for i in range(0, len(lst), n)]

    return zl
    
  
  # def noise(self,next_action):
  #   for i in range(len(next_action)):
  #       for j in range(len(next_action[i])):
  #           noise = torch.zeros_like(next_action[i][0])
  #           # print("noise = ", next_action[i][j])
  #           next_action[i][j].tolist()
  #           rand = random.sample(range(0, env.ANNum+env.CNNum), 2)         
  #           # if j[rand[0]] != 0:
  #           red = next_action[i][j][rand[0]] *20/100
  #           noise[rand[0]] = -red
  #           noise[rand[1]] = red
  #           next_action[i][j] = next_action[i][j]+noise
      
  #   return next_action

  def reform(self, next_action):
      nex = []
      for i in next_action:
          nex.append(i.cpu().data.numpy())
      # print("next : ", nex)    
      act = [[] for i in range(batch_size)]
      for i in range(len(act)):
          for j in range(len(nex)):
              for k in range(len(nex[j])):
                  if i == k:
                      act[i].append(nex[j][k].tolist())
                      
      # print("act : ", act)
      act = np.array(act)
      act1 = [[] for i in range(batch_size)]
      for i in range(len(act)):
          # print("debug -->", act[i].flatten())
          act1[i] = act[i].flatten()
      return act1
      
        
  def train(self, replay_buffer, iterations, batch_size=10, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      # next_action = self.actor_target(next_state)
      y1,y2,y3,y4,y5,y6,y7,y8,y9 = self.actor_target(next_state)
      next_action = torch.cat([y1,y2,y3,y4,y5,y6,y7,y8,y9],1)
    
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      
      # next_action = self.noise(next_action)
      
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      
      # act1 = self.reform(next_action)            
      # next_action = torch.Tensor(act1).to(device)
      # print(next_state)
      # print(next_action)
      
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      # because the target Q is still in the computation graph, then it need to be dettached inorder to add with the reward
        
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      #########################################################################################################################
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        y1,y2,y3,y4,y5,y6,y7,y8,y9 = self.actor(state)
        action = torch.cat([y1,y2,y3,y4,y5,y6,y7,y8,y9],1)
        actor_loss = -self.critic.Q1(state,action).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


# ## We make a function that evaluates the policy by calculating its average reward over 10 episodes

# In[7]:
rew =[]

def evaluate_policy(policy, eval_episodes=10):
  indexx = 1
  avg_reward = 0.
  div_avg = eval_episodes * 10
  for _ in range(eval_episodes):
    # obs = env.reset()
    
    obs = env.get_observation()[2]
    # print("obs = ", obs)
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      # print("action==== ", action)
      obs, reward, done = env.step(action)
       # print("Done==== ", env.lamda)
      print("reward:", reward)
      avg_reward += reward
      if indexx%10 ==0:
          rew.append(reward)
      # print("reward =", avg_reward)
      # rew.append(reward)
      indexx+=1
    print("---------------")
  avg_reward /= div_avg
  
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward


# ## We set the parameters

# In[8]:


# env_name = "HalfCheetahBulletEnv-v0" # Name of a environment (set it to any Continous environment you want)
seed = 0 # Random seed number
start_timesteps = 100  # 2000 1e4 Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 100   #   1000 5e3 How often the evaluation step is performed (after how many timesteps)
max_timesteps = 10000  # 12000  5e5number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 10 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated


# ## We create a file name for the two saved models: the Actor and Critic models

# In[9]:


file_name = "%s_%s_%s" % ("TD3", "MEC", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")


# ## We create a folder inside which will be saved the trained models

# In[10]:


if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")


# ## We create the PyBullet environment

# In[11]:


env = EnvironmentTD3()
print(env.lamda)


# ## We set seeds and we get the necessary information on the states and actions in the chosen environment

# In[12]:


state_dim = env.get_observation()[2].shape[0]
print("state_dim = ", state_dim)
action_dim = env.ANNum+env.CNNum
action_dim_critic = (env.ANNum+env.CNNum)*env.ANNum*env.CNNum
print("action_dim = ",action_dim)
print("action_dim_critic = ", action_dim_critic)
# max_action = float(env.get_actions().high[0])


# ## We create the policy network (the Actor model)

# In[13]:


policy = TD3(state_dim, action_dim, action_dim_critic)


# ## We create the Experience Replay memory

# In[14]:


replay_buffer = ReplayBuffer()


# ## We define a list where all the evaluation results over 10 episodes are stored

# In[15]:


evaluations = [evaluate_policy(policy)]
print("ini:", evaluations)
print(rew)
# ## We create a new folder directory in which the final results (videos of the agent) will be populated

# In[16]:


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')
max_episode_steps = 10

# ## We initialize the variables

# In[17]:


total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()


# ## Training

# In[17]:
import timeit    
start = timeit.default_timer()

def noise(data):
    print("data : ", data)
    ind =0                                                                               
    for a in data:                                                                                
        # print("a = ", a)
        index=0
        for i in a:
            # print ("i = ",i)
            # r = random.randint(0,5)
            r = np.argmax(data[ind][index])
            s = random.randint(0,env.ANNum)
            # t = random.randint(0,5)
            t = np.argmax(data[ind][index])
            u = random.randint(0,env.ANNum)
            x = np.random.uniform(0.0,0.5)
            y= np.random.uniform(0.0,0.5)
            red = data[ind][index][r]*x
            # print("c[index][r] = ", c[index])
            data[ind][index][r] =  data[ind][index][r] - red
            data[ind][index][s] = data[ind][index][s] + red
            red = data[ind][index][t]*y
            data[ind][index][t] =  data[ind][index][t] - red
            data[ind][index][u] = data[ind][index][u] +red
            index+=1
        # x = np.array(c).flatten()
        # data[ind]= x
        ind += 1
    return data
# count = 0
# We start the main loop over 500,000 timesteps
while total_timesteps < max_timesteps:
  lst_reward =[]
  # If the episode is done
  if done: 

    # If we are not at the very beginning, we start the training process of the model
    if total_timesteps != 0:
      print("++++++++++++++++++++++ Training Process ++++++++++++++++++++++++")
      print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
      policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
      
    # We evaluate the episode and we save the policy
    if timesteps_since_eval >= eval_freq:
      timesteps_since_eval %= eval_freq
      evaluations.append(evaluate_policy(policy))
      print("evaluations =", evaluations)
      policy.save(file_name, directory="./pytorch_models")
      np.save("./results/%s" % (file_name), evaluations)
    
    # When the training step is done, we reset the state of the environment
    obs = env.reset()
    obs1, obs2, obs = env.get_observation()
    # Set the Done to False
    done = False
    
    # Set rewards and episode timesteps to zero
    episode_reward = 0
    episode_timesteps = 0
    episode_num += 1
  
  # Before 2000 timesteps, we play random actions
  if total_timesteps < start_timesteps:
    array, action = env.get_actions()
    action = array
    # print("==== Random Action ====")
    
  else: # After 10000 timesteps, we switch to the model
    action = policy.select_action(np.array(obs))
    # print("action: ", action)
    print("==== Neural Network Action ====")
    # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    if expl_noise != 0:
      action = noise(action)
      
  # print("=====================OBS============================", obs)
  # The agent performs the action in the environment, then reaches the next state and receives the reward
  new_obs, reward, done = env.step(action)
  # print("=====================new_obs============================", new_obs)
  # print((obs==new_obs).all())
  # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
  # obsa, obsb, obsc = env.get_observation()
  # print("obs 2 = ", env.AN)
  # print("obs 2 = ", env.CN)
  # We check if the episode is done
  done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
  # print("step reward = ", reward)
  # We increase the total reward
  episode_reward += reward
 
  # We store the new transition into the Experience Replay memory (ReplayBuffer)
  
  # print("action= ", action)
  zx = []
  for i in action:
      zx.append(np.concatenate(i).ravel())
  action = np.array(zx).flatten()
  # print("action= ", action)
  # xc
  # print("action= ", action.dtype)
  # print("+++++++++++++++++++++++++++++++++++ Total time spet:", total_timesteps)
  replay_buffer.add((obs, new_obs, action, reward, done_bool))

  # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
  obs = new_obs
  episode_timesteps += 1
  total_timesteps += 1
  timesteps_since_eval += 1
# print("One Episode Reward = ", episode_reward)
# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy))
print("evaluations Final ===========", evaluations)
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
stop = timeit.default_timer()
execution_time = stop - start
ev_reward = evaluate_policy(policy)

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

with open("train-4000-5okt.txt", "a+") as file_object:
    # Move read cursor to the start of file.
    file_object.seek(0)
    # If file is not empty then append '\n'
    data = file_object.read(100)
    if len(data) > 0 :
        file_object.write("\n")
    # Append text at the end of file
    file_object.write("\n")
    file_object.write("############")
    file_object.write(dt_string)
    file_object.write("#############")
    file_object.write("\n")
    file_object.write("Evaluation reward = ")
    file_object.write(str(ev_reward))
    file_object.write("\n")
    file_object.write("Program Executed in ")
    file_object.write(str(execution_time))
    file_object.write("\n")
    file_object.write("CN Num = ")
    file_object.write(str(env.CNNum))
    file_object.write("\n")
    file_object.write("Traffic = ")
    file_object.write(str(env.lam))
    file_object.write("\n")
    file_object.write("evaluations = ")
    file_object.write(str(evaluations))
    file_object.write("\n")
    file_object.write("rew = ")
    file_object.write(str(rew))
    file_object.write("\n")
print("Training done in "+str(execution_time))
print("rew", rew)
print()

# ## Inference

# In[18]:

# class Actor(nn.Module):
  
#   def __init__(self, state_dim, action_dim):
#     super(Actor, self).__init__()
#     self.layer_1 = nn.Linear(state_dim, 84)
#     self.layer_2 = nn.Linear(84, 42)
#     self.layer_3 = nn.Linear(42, 42)
    
#     self.layer_11 = nn.Linear(42, 84)
#     self.layer_12 = nn.Linear(84, 42)
#     self.layer_13 = nn.Linear(42, action_dim)
#     self.layer_21 = nn.Linear(42, 84)
#     self.layer_22 = nn.Linear(84, 42)
#     self.layer_23 = nn.Linear(42, action_dim)
#     self.layer_31 = nn.Linear(42, 84)
#     self.layer_32 = nn.Linear(84, 42)
#     self.layer_33 = nn.Linear(42, action_dim)
#     self.layer_41 = nn.Linear(42, 84)
#     self.layer_42 = nn.Linear(84, 42)
#     self.layer_43 = nn.Linear(42, action_dim) 
#     self.layer_51 = nn.Linear(42, 84)
#     self.layer_52 = nn.Linear(84, 42)
#     self.layer_53 = nn.Linear(42, action_dim)
#     self.layer_61 = nn.Linear(42, 84)
#     self.layer_62 = nn.Linear(84, 42)
#     self.layer_63 = nn.Linear(42, action_dim)
#     self.layer_71 = nn.Linear(42, 84)
#     self.layer_72 = nn.Linear(84, 42)
#     self.layer_73 = nn.Linear(42, action_dim)
#     self.layer_81 = nn.Linear(42, 84)
#     self.layer_82 = nn.Linear(84, 42)
#     self.layer_83 = nn.Linear(42, action_dim)
#     self.layer_91 = nn.Linear(42, 84)
#     self.layer_92 = nn.Linear(84, 42)
#     self.layer_93 = nn.Linear(42, action_dim)
    
   

#   def forward(self, x): #x is the input state
#     x = F.relu(self.layer_1(x))
#     x = F.relu(self.layer_2(x))
#     x = F.relu(self.layer_3(x))
#     ####################################################################################
#     y1 = F.relu(self.layer_11(x))
#     y1 = F.relu(self.layer_12(y1))
#     # y1 = self.layer_13(y1)
#     y1 = F.softmax(self.layer_13(y1),dim=1)
    
#     y2 = F.relu(self.layer_21(x))
#     y2 = F.relu(self.layer_22(y2))
#     # y2 = self.layer_23(y2)
#     y2 = F.softmax(self.layer_23(y2),dim=1)
     
#     y3 = F.relu(self.layer_31(x))
#     y3 = F.relu(self.layer_32(y3))
#     # y3 = self.layer_33(y3)
#     y3 = F.softmax(self.layer_33(y3),dim=1)
    
#     y4 = F.relu(self.layer_41(x))
#     y4 = F.relu(self.layer_42(y4))
#     # y4 = self.layer_43(y4)
#     y4 = F.softmax(self.layer_43(y4),dim=1)
    
#     y5 = F.relu(self.layer_51(x))
#     y5 = F.relu(self.layer_52(y5))
#     # y5 = self.layer_53(y5)
#     y5 = F.softmax(self.layer_53(y5),dim=1)
    
#     y6 = F.relu(self.layer_61(x))
#     y6 = F.relu(self.layer_62(y6))
#     # y6 = self.layer_63(y6)
#     y6 = F.softmax(self.layer_63(y6),dim=1)
     
#     y7 = F.relu(self.layer_71(x))
#     y7 = F.relu(self.layer_72(y7))
#     # y7 = self.layer_73(y7)
#     y7 = F.softmax(self.layer_73(y7),dim=1)
    
#     y8 = F.relu(self.layer_81(x))
#     y8 = F.relu(self.layer_82(y8))
#     # y8 = self.layer_83(y8)
#     y8 = F.softmax(self.layer_83(y8),dim=1)
#     # print("y1", y1)
#     y9 = F.relu(self.layer_91(x))
#     y9 = F.relu(self.layer_92(y9))
#     # y5 = self.layer_53(y5)
#     y9 = F.softmax(self.layer_93(y9),dim=1)
   
    
#     return y1,y2,y3,y4,y5,y6,y7,y8,y9


# class Critic(nn.Module):
  
#   def __init__(self, state_dim, action_dim_critic):
#     super(Critic, self).__init__()
#     self.layer_1 = nn.Linear(state_dim + action_dim_critic, 134)
#     self.layer_2 = nn.Linear(134, 67)
#     self.layer_3 = nn.Linear(67, 1)
#     # Defining the second Critic neural network
#     self.layer_4 = nn.Linear(state_dim + action_dim_critic, 134)
#     self.layer_5 = nn.Linear(134, 67)
#     self.layer_6 = nn.Linear(67, 1)

#   def forward(self, x, u): #x is state, u is action
#     xu = torch.cat([x, u], 1) #vertical concantecated 
#     # Forward-Propagation on the first Critic Neural Network
#     x1 = F.relu(self.layer_1(xu))
#     x1 = F.relu(self.layer_2(x1))
#     x1 = self.layer_3(x1)
#     # Forward-Propagation on the second Critic Neural Network
#     x2 = F.relu(self.layer_4(xu))
#     x2 = F.relu(self.layer_5(x2))
#     x2 = self.layer_6(x2)
#     return x1, x2

#   def Q1(self, x, u):
#     xu = torch.cat([x, u], 1)
#     x1 = F.relu(self.layer_1(xu))
#     x1 = F.relu(self.layer_2(x1))
#     x1 = self.layer_3(x1)
#     return x1


# # Selecting the device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # Building the whole Training Process into a class

# class TD3(object):
  
#   def __init__(self, state_dim, action_dim, action_dim_critic):
#     self.actor = Actor(state_dim, action_dim).to(device)
#     self.actor_target = Actor(state_dim, action_dim).to(device)
#     self.actor_target.load_state_dict(self.actor.state_dict())
#     self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
#     self.critic = Critic(state_dim, action_dim_critic).to(device)
#     self.critic_target = Critic(state_dim, action_dim_critic).to(device)
#     self.critic_target.load_state_dict(self.critic.state_dict())
#     self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

#   def select_action(self, state):
#     state = torch.Tensor(state.reshape(1, -1)).to(device)
#     xl = self.actor(state)
#     lst = []
#     n = env.ANNum
#     for i in xl:
#         lst.append(i.cpu().data.numpy().flatten())
#     zl = [lst[i:i + n] for i in range(0, len(lst), n)]
#     # print("yl = ",zl)
#     # print("##########################################################################")
#     return zl
    

#   def noise(self, data):
#     ind =0                                                                               
#     for a in data:                                                                                
#         b = np.array_split(a,8)
#         c = b
#         index=0
#         for i in c:
#             # print (i)
#             r = random.randint(0,5)
#             s = random.randint(0,5)
#             t = random.randint(0,5)
#             u = random.randint(0,5)
        
#             red = b[index][r]*20/100
#             c[index][r] =  b[index][r] - red
#             c[index][s] = b[index][s] + red
#             red = b[index][t]*20/100
#             c[index][t] =  b[index][t] - red
#             c[index][u] = b[index][u] +red
#             index+=1
#         x = np.array(c).flatten()
#         data[ind]= x
#         ind += 1
#     return data

#   def reform(self, next_action):
#       nex = []
#       for i in next_action:
#           nex.append(i.cpu().data.numpy())
#       # print("next : ", nex)    
#       act = [[] for i in range(batch_size)]
#       for i in range(len(act)):
#           for j in range(len(nex)):
#               for k in range(len(nex[j])):
#                   if i == k:
#                       act[i].append(nex[j][k].tolist())
                      
#       # print("act : ", act)
#       act = np.array(act)
#       act1 = [[] for i in range(batch_size)]
#       for i in range(len(act)):
#           # print("debug -->", act[i].flatten())
#           act1[i] = act[i].flatten()
#       return act1
      
        
#   def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=4):
    
#     for it in range(iterations):
      
#       # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
#       batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
#       state = torch.Tensor(batch_states).to(device)
#       next_state = torch.Tensor(batch_next_states).to(device)
#       action = torch.Tensor(batch_actions).to(device)
#       reward = torch.Tensor(batch_rewards).to(device)
#       done = torch.Tensor(batch_dones).to(device)
      
#       # print("batch_states: ", batch_states[0])
#       # print("batch_next_states: ", batch_next_states[0])
#       # print("batch_actions: ", batch_actions[0])
#       # print("batch_rewards: ", batch_rewards[0])
#       # print("batch_dones: ", batch_dones[0])
#       # Step 5: From the next state s’, the Actor target plays the next action a’
#       y1,y2,y3,y4,y5,y6,y7,y8,y9 = self.actor_target(next_state)
#       next_action = torch.cat([y1,y2,y3,y4,y5,y6,y7,y8,y9],1)
      
#       # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
#       # noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
#       # noise = noise.clamp(-noise_clip, noise_clip)
#       # next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
#       # next_action = self.noise(next_action)
      
#       # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      
#       # act1 = self.reform(next_action)            
#       # next_action = torch.Tensor(act1).to(device)
     
      
#       target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
#       # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
#       target_Q = torch.min(target_Q1, target_Q2)
      
#       # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
#       target_Q = reward + ((1 - done) * discount * target_Q).detach()
#       # because the target Q is still in the computation graph, then it need to be dettached inorder to add with the reward
        
#       # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      
#       current_Q1, current_Q2 = self.critic(state, action)
      
#       # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
#       critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
#       # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
#       self.critic_optimizer.zero_grad()
#       critic_loss.backward()
#       self.critic_optimizer.step()
      
#       #########################################################################################################################
      
#       # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
#       if it % policy_freq == 0:
        
#         # print("state = ", state)
#         # print("state s = ", state.size())
#         y1,y2,y3,y4,y5,y6,y7,y8,y9 = self.actor(state)
#         action1 = torch.cat([y1,y2,y3,y4,y5,y6,y7,y8,y9],1)
#         action = torch.Tensor(action1).to(device)
#         # print("action1 = ", action)
#         # print("action1 s = ", action.size())
#         actor_loss = -self.critic.Q1(state,action).mean()
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
#         for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#           target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
#         # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
#         for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#           target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
#   # Making a save method to save a trained model
#   def save(self, filename, directory):
#     torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
#     torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
#   # Making a load method to load a pre-trained model
#   def load(self, filename, directory):
#     self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
#     self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
   
# import timeit    
# def evaluate_policy(policy, eval_episodes=10):
#   div_avg = eval_episodes * 10
#   avg_reward = 0.
#   ex_time=[]
#   for _ in range(eval_episodes):
#     obs = env.reset()
#     obs = env.get_observation()[2]
#     done = False
#     # index = 0
#     while not done:
#       start = timeit.default_timer()
#       action = policy.select_action(np.array(obs))
#       # index +=1 
#       # print(index)
#       stop = timeit.default_timer()
#       # execution_time = stop - start
#       ex_time.append((stop - start))
#       print("Program Executed in "+str(stop - start))
#       # print("action: ", action)
#       obs, reward, done = env.step(action)
#       # print("env AN: ", env.AN)
#       # print("env CN: ", env.CN)
      
#       # print("Done==== ", done)
#       avg_reward += reward
#       # print("reward =", avg_reward)
#   avg_reward /= div_avg
#   print ("---------------------------------------")
#   print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
#   print ("---------------------------------------")
#   from datetime import datetime
#   now = datetime.now()
#   dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#   with open("result.txt", "a+") as file_object:
#     # Move read cursor to the start of file.
#     file_object.seek(0)
#     # If file is not empty then append '\n'
#     data = file_object.read(100)
#     if len(data) > 0 :
#         file_object.write("\n")
#     # Append text at the end of file
#     file_object.write("\n")
#     file_object.write("############")
#     file_object.write(dt_string)
#     file_object.write("#############")
#     file_object.write("\n")
#     file_object.write("Total reward = ")
#     file_object.write(str(avg_reward))
#     file_object.write("\n")
#     file_object.write("Program Executed in ")
#     file_object.write(str(sum(ex_time)/len(ex_time)))
#     file_object.write("\n")
#     file_object.write("Traffic = ")
#     file_object.write(str(env.lam))
#     file_object.write("\n")
#     file_object.write("Action_dim_critic = ")
#     file_object.write(str(action_dim_critic))
#     file_object.write("\n")
#     file_object.write("State_dim = ")
#     file_object.write(str(state_dim))
#     file_object.write("\n")
#   return avg_reward



# # env_name = "HalfCheetahBulletEnv-v0"
# seed = 0

# file_name = "%s_%s_%s" % ("TD3", "MEC", str(seed))
# print ("---------------------------------------")
# print ("Settings: %s" % (file_name))
# print ("---------------------------------------")

# eval_episodes = 10
# save_env_vid = False
# env = EnvironmentTD3()
# max_episode_steps = 10
# # print("env AN: ", env.AN)
# # print("env CN: ", env.CN)
# # if save_env_vid:
# #   env = wrappers.Monitor(env, monitor_dir, force = True)
# #   env.reset()
# # env.seed(seed)
# # torch.manual_seed(seed)
# # np.random.seed(seed)
# state_dim = env.get_observation()[2].shape[0]
# action_dim = env.ANNum+env.CNNum
# action_dim_critic = (env.ANNum+env.CNNum)*env.ANNum*env.CNNum
# policy = TD3(state_dim, action_dim, action_dim_critic)
# policy.load(file_name, './pytorch_models/')
# _ = evaluate_policy(policy, eval_episodes=eval_episodes)


# In[19]:


# print(gym.__version__)
# 

# In[ ]:




