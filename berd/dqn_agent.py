import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden=[128,128], n_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, device, lr=1e-3, gamma=0.99, batch_size=64, target_update=1000):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer()
        self.step_count = 0

    def select_action(self, state_batch, epsilon):
        # state_batch: torch.tensor [N, state_dim]
        if random.random() < epsilon:
            # random for each env
            batch_size = state_batch.shape[0]
            return torch.randint(0, 2, (batch_size,), device=self.device)
        else:
            with torch.no_grad():
                qvals = self.policy_net(state_batch)
                actions = qvals.argmax(dim=1)
                return actions

    def store_transition(self, state, action, reward, next_state, done):
        # states as numpy arrays or tuples
        self.replay.push(np.array(state, dtype=np.float32),
                         int(action),
                         float(reward),
                         np.array(next_state, dtype=np.float32),
                         bool(done))

    def optimize(self):
        if len(self.replay) < self.batch_size:
            return None

        trans = self.replay.sample(self.batch_size)
        state = torch.tensor(np.stack(trans.state), dtype=torch.float32, device=self.device)
        action = torch.tensor(trans.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(trans.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(np.stack(trans.next_state), dtype=torch.float32, device=self.device)
        done = torch.tensor(trans.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q
        q_values = self.policy_net(state).gather(1, action)

        # target Q
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            target_q = reward + (1.0 - done) * (self.gamma * next_q)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # optional gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
