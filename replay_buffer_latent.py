import numpy as np
import torch

class ReplayBufferLatent(object):
    def __init__(self, state_dim, action_dim, latent_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.latent = np.zeros((max_size, latent_dim))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add(self, state, action, next_state, reward, done, latent):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.latent[self.ptr] = latent

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        # print(ind)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.latent[ind]).to(self.device)
        )

    def get_all(self):
        return (
            torch.FloatTensor(self.state).to(self.device),
            torch.FloatTensor(self.action).to(self.device),
            torch.FloatTensor(self.next_state).to(self.device),
            torch.FloatTensor(self.reward).to(self.device),
            torch.FloatTensor(self.not_done).to(self.device),
            torch.FloatTensor(self.latent).to(self.device)
        )

    def reset(self):
        self.ptr = 0
        self.size = 0

        # self.state = np.zeros((self.max_size, self.state_dim))
        # self.action = np.zeros((self.max_size, self.action_dim))
        # self.next_state = np.zeros((self.max_size, self.state_dim))
        # self.reward = np.zeros((self.max_size, 1))
        # self.not_done = np.zeros((self.max_size, 1))
        # self.latent = np.zeros((self.max_size, self.latent_dim))
