import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, latent_cont_dim=0, latent_disc_dim=0, hidden=(256,256), l2_max=100):
        super(Actor, self).__init__()

        l2_dim = np.minimum(state_dim, l2_max)

        self.l_latent = nn.Linear(latent_cont_dim + latent_disc_dim, l2_dim)
        self.l1 = nn.Linear(state_dim + l2_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], action_dim)

        self.max_action = max_action

    def forward(self, state, latent):
        z = F.relu(self.l_latent(latent))

        sz = torch.cat([state, z], 1)
        a = F.relu(self.l1(sz))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, latent_cont_dim=0, latent_disc_dim=0, hidden=(256,256), l2_max=100):
        super(Critic, self).__init__()

        latent_dim = latent_cont_dim + latent_disc_dim
        l2_dim = np.minimum(state_dim, l2_max)

        # Q1 architecture
        self.l_latent1 = nn.Linear(latent_dim, l2_dim)
        self.l1 = nn.Linear(state_dim + action_dim + l2_dim, hidden[0])

        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.l3 = nn.Linear(hidden[1], 1)
        self.l3_z = nn.Linear(hidden[1], latent_dim)

        # Q2 architecture
        self.l_latent2 = nn.Linear(latent_dim, l2_dim)
        self.l4 = nn.Linear(state_dim + action_dim + l2_dim, hidden[0])

        self.l5 = nn.Linear(hidden[0], hidden[1])
        self.l6 = nn.Linear(hidden[1], 1)
        self.l6_z = nn.Linear(hidden[1], latent_dim)

    def forward(self, state, action, latent):
        z1 = F.relu(self.l_latent1(latent))

        saz1 = torch.cat([state, action, z1], 1)

        h1 = F.relu(self.l1(saz1))
        h1 = F.relu(self.l2(h1))
        q1 = self.l3(h1)

        z2 = F.relu(self.l_latent2(latent))

        saz2 = torch.cat([state, action, z2], 1)

        h2 = F.relu(self.l4(saz2))
        h2 = F.relu(self.l5(h2))
        q2 = self.l6(h2)

        return q1, q2

    def Q1(self, state, action, latent):
        z1 = F.relu(self.l_latent1(latent))
        saz1 = torch.cat([state, action, z1], 1)

        q1 = F.relu(self.l1(saz1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, latent_cont_dim=0, latent_disc_dim=0, hidden=(256,256)):
        super(Discriminator, self).__init__()

        # Z1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])

        if not latent_cont_dim == 0:
            self.l3_z_cont = nn.Linear(hidden[1], latent_cont_dim)

        if not latent_disc_dim == 0:
            self.l3_z_disc = nn.Linear(hidden[1], latent_disc_dim)

        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        z_cont = None
        z_disc = None
        h = F.relu(self.l1(sa))
        h = F.relu(self.l2(h))
        if not self.latent_cont_dim == 0:
            z_cont = self.l3_z_cont(h)
        if not self.latent_disc_dim == 0:
            z_disc = F.softmax(self.l3_z_disc(h))

        return z_cont, z_disc


class LTD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            latent_cont_dim = 1,
            latent_disc_dim = 2,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            info_freq=4,
            iw = 'IW',
            clip_ratio=0.2,
            elite_ratio = 3,
            hidden=(256,256)
    ):

        latent_dim = latent_cont_dim + latent_disc_dim

        self.actor = Actor(state_dim, action_dim, max_action, latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim, hidden=hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim, hidden=hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.discriminator = Discriminator(state_dim, action_dim, latent_cont_dim=latent_cont_dim, latent_disc_dim=latent_disc_dim, hidden=hidden).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, betas=(0.5, 0.999))
        self.info_optimizer = torch.optim.Adam( itertools.chain(self.actor.parameters(),
                                                                self.discriminator.parameters()), lr=3e-4, betas=(0.5, 0.999))

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.info_freq = info_freq
        self.latent_dim = latent_dim
        self.latent_cont_dim = latent_cont_dim
        self.latent_disc_dim = latent_disc_dim
        self.iw = iw
        self.clip_ratio = clip_ratio
        self.elite_ratio = elite_ratio

        self.total_it = 0

    def select_action(self, state, z):
        z = torch.FloatTensor(z.reshape(1, -1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state, z).cpu().data.numpy().flatten()

        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, latent = replay_buffer.sample(batch_size)

        with torch.no_grad():

            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            # z_next = torch.FloatTensor( np.random.normal(size=(batch_size, self.latent_dim)) ).to(device)
            # z_next = torch.FloatTensor(np.random.uniform(-1, 1, size=(batch_size, self.latent_dim))).to(device)
            next_action = (
                    self.actor_target(next_state, latent) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action, latent)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, latent)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            a = self.actor(state, latent)
            actor_loss = - self.critic.Q1(state, a, latent).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.total_it % self.info_freq == 0:

            if self.iw == 'IW':
                state, action, next_state, reward, not_done, latent = replay_buffer.sample(batch_size)

                self.info_optimizer.zero_grad()

                action = self.actor(state, latent)
                z_cont, z_disc = self.discriminator(state, action)

                with torch.no_grad():
                    current_Q1, current_Q2 = self.critic_target(state, action, latent)
                    minQ = torch.min(current_Q1, current_Q2)

                    max_minQ = torch.max(minQ)
                    weight = torch.exp(minQ - max_minQ) / torch.mean(torch.exp(minQ - max_minQ))
                    weight_clip = torch.clamp(weight, 1 - self.clip_ratio, 1 + self.clip_ratio)

                info_loss = 0
                disc_loss = nn.CrossEntropyLoss(reduction='none')
                if not self.latent_cont_dim == 0:
                    latent_cont = None
                    if self.latent_disc_dim == 0:
                        latent_cont = latent
                    else:
                        latent_cont = latent[:, 0:self.latent_cont_dim]

                    info_loss += torch.mean(weight_clip.detach() * F.mse_loss(z_cont, latent_cont.detach(), reduction='none'))


                if not self.latent_disc_dim == 0:
                    latent_disc = None
                    if self.latent_cont_dim == 0:
                        latent_disc = latent
                    else:
                        latent_disc = latent[:, self.latent_cont_dim:self.latent_dim]

                    latent_disc_label = torch.argmax(latent_disc, dim=1)
                    info_loss += torch.mean(weight_clip.detach() * disc_loss(z_disc, latent_disc_label.detach()).view(-1,1))

                info_loss.backward()
                self.info_optimizer.step()


            else:
                state, action, next_state, reward, not_done, latent = replay_buffer.sample(batch_size)

                self.info_optimizer.zero_grad()

                action = self.actor(state, latent)
                z_cont, z_disc = self.discriminator(state, action)

                info_loss = 0
                if not self.latent_cont_dim == 0:
                    latent_cont = None
                    if self.latent_disc_dim == 0:
                        latent_cont = latent
                    else:
                        latent_cont = latent[:, 0:self.latent_cont_dim]
                    info_loss += F.mse_loss(z_cont, latent_cont)

                if not self.latent_disc_dim == 0:
                    latent_disc = None
                    if self.latent_cont_dim == 0:
                        latent_disc = latent
                    else:
                        latent_disc = latent[:, self.latent_cont_dim:self.latent_dim]

                    latent_disc_label = torch.argmax(latent_disc, dim=1)
                    info_loss += F.cross_entropy(z_disc, latent_disc_label)

                info_loss.backward()
                self.info_optimizer.step()

    def save_model(self, iter, seed, env_name, args, foldername='./model/ltd3'  ):
        try:
            import pathlib
            pathlib.Path(foldername).mkdir(parents=True, exist_ok=True)

            IW = None
            if args['IW'] == 'IW':
                IW = 'IW_' + str(args['iw_clip_ratio'])
            else:
                IW = args['IW']

            torch.save(self.actor.state_dict(),
                       foldername + '/ltd3_actor_'+ env_name
                       + '_cont' + str(self.latent_cont_dim) + '_disc' + str(self.latent_disc_dim) + '_' + IW
                       + '_policy_freq_' + str(int(args['policy_freq']))  + '_info_freq' + str(self.info_freq) + '_hidden' + str(args['hidden'])
                       + '_seed' + str(seed) + '_iter' + str(iter) + '.pth')

            print('models is saved for iteration', iter )

        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

    def load_model(self, iter, seed, env_name, args, foldername='model/ltd3'  ):
        IW = None
        if args['IW'] == 'IW':
            IW = 'IW_' + str(args['iw_clip_ratio'])
        else:
            IW = args['IW']

        self.actor.load_state_dict(torch.load(
            foldername + '/ltd3_actor_' + env_name
            + '_cont' + str(self.latent_cont_dim) + '_disc' + str(self.latent_disc_dim) + '_' + IW
            + '_policy_freq_' + str(int(args['policy_freq']))  + '_info_freq' + str(self.info_freq) + '_hidden' + str(args['hidden']) +\
            '_seed' + str(seed) + '_iter' + str(iter) + '.pth'))

