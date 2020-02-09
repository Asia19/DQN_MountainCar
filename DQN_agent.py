import gym
import torch
import random
import copy
import numpy as np
import torch.nn.functional as F
from collections import namedtuple, deque
from torch import nn
from gym import make

BUFFER_SIZE = 20000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 0.001
C = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(result)


class DQN_nn(nn.Module):
    def __init__(self, state_dim, action_dim, seed, hidden=63):
        super(DQN_nn, self).__init__()
        self.seed = random.seed(seed)
        self.hidden = hidden
        
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class Agent:
    def __init__(self, state_dim, action_dim, seed, lr=LR):
        def init_weights(layer):
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = GAMMA
        self.Q = DQN_nn(state_dim, action_dim, seed)
        self.Q.apply(init_weights)
        self.TQ = copy.deepcopy(self.Q)
        self.Q.to(device)
        self.TQ.to(device)
        self.buffer = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.t_step = 0
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad():
            Q_value = self.Q(state)
        self.Q.train()
        return torch.argmax(Q_value).item()        
    
    def step(self, transition):
        self.t_step = (self.t_step + 1) % C
        self.buffer.add(transition)
        if len(self.buffer) > BATCH_SIZE:
            train_batch = self.buffer.sample()
            self.train(train_batch)
        if self.t_step == 0:
            self.TQ = copy.deepcopy(self.Q)        
        
    def train(self, train_batch):
        states, actions, rewards, next_states, dones = train_batch
        
        Q = self.Q(states).gather(1, actions)
        Q1 = self.TQ(next_states)
        maxQ1 = torch.max(Q1, -1)[0].unsqueeze(1)
        TQ = rewards + (self.gamma * maxQ1.detach() * (1 - dones))
        
        loss = self.loss(Q, TQ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path="agent.pkl"):
        torch.save(self.Q.state_dict(), path)
        

class ReplayBuffer:
    """we store a fixed amount of the last transitions <f_t, a_t, r_t, f_t+1>"""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, transition):
        e = self.experience(*transition)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)