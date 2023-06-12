import numpy as np
import torch
import torch.nn as nn
import gym
from torch.optim import Adam
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.signal import lfilter
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./tb_record_PPO")

Transition = namedtuple(
    'Transition', ('state', 'action', 'log_prob', 'next_state', 'reward', 'value'))

SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        self.Net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(0.1),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)

    def forward(self, inputs):
        policys = self.Net(inputs)
        return policys


class Network(nn.Module):
    def __init__(self, nums_state, hidden_size, nums_action):
        super().__init__()
        self.Critic = MLP(nums_state, hidden_size, 1)
        self.Policy = MLP(nums_state, hidden_size, nums_action)

class PEMO(object):
    def __init__(self, num_states, num_actions, hidden_size, discount, clip_r, STEP):
        self.num_actions = num_actions
        self.num_states = num_states
        self.hidden_size = hidden_size
        self.discount = discount
        self.clip_r = clip_r
        self.Normalize = True
        self.STEP = STEP
        self.Network = Network(num_states, hidden_size, num_actions)
        self.Network_optim = Adam([{'params': self.Network.Critic.parameters()},
                                   {'params': self.Network.Policy.parameters()}], lr=5e-4)

    '''
    Sample trajectory
    '''
    def get_info_from_env(self):
        
        Trajectory = []

        state, _ = train_env.reset()
        state = torch.FloatTensor(state)

        train_reward = 0

        while True:
            
            action_dist = self.Network.Policy(state)

            value = self.Network.Critic(state)

            action_prob = F.softmax(action_dist, dim = -1)
            
            action_categorical = Categorical(action_prob)

            action = action_categorical.sample()

            log_prob = action_categorical.log_prob(action)
            
            next_state, reward, done, truncated, _ = train_env.step(action.detach().numpy())
            
            train_reward += reward

            next_state = torch.FloatTensor(next_state)

            reward = torch.FloatTensor([reward])
            
            Trajectory.append(Transition(state, action, log_prob, next_state, reward, value))

            state = next_state
            
            if done or truncated:
                break
        return Transition(*zip(*Trajectory)), train_reward
    
    
    def get_training_data(self):

        trajectory, train_reward = self.get_info_from_env()
        
        reward = torch.Tensor(trajectory.reward)
        state = torch.stack(list(trajectory.state), dim=0)
        log_prob = torch.Tensor(trajectory.log_prob)
        action = torch.Tensor(trajectory.action)
        value = torch.Tensor(trajectory.value)
        returns = torch.Tensor(lfilter([1], [1, -self.discount], list(reward)[::-1])[::-1].copy())

        if self.Normalize:
            
            returns = (returns - torch.mean(returns)) / torch.std(returns)

        advantages = returns - value

        if self.Normalize:

            advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        return train_reward, state, action.unsqueeze(1), log_prob.unsqueeze(1), returns.unsqueeze(1), advantages.unsqueeze(1)

    def update(self, state, action, log_prob, returns, advantages):
        
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.STEP):

            new_action_dist = self.Network.Policy(state)

            value_free = self.Network.Critic(state)

            new_action_prob = F.softmax(new_action_dist, dim = -1)

            new_action_categorical = Categorical(new_action_prob)

            new_log_prob = new_action_categorical.log_prob(action.squeeze(1)).unsqueeze(1)
            
            # kl_diverge = torch.exp(log_prob) * (log_prob - new_log_prob) * 0.01

            ratio = torch.exp(new_log_prob - log_prob)

            policy_loss_1 = ratio * advantages
           
            policy_loss_2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages

            # policy_loss = - torch.mean(torch.min(policy_loss_1, policy_loss_2) - kl_diverge)

            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

            value_loss = F.smooth_l1_loss(returns, value_free).mean()

            total_loss = policy_loss + value_loss

            self.Network_optim.zero_grad()

            total_loss.backward()

            self.Network_optim.step()

            total_policy_loss += policy_loss.detach().item()
            total_value_loss += value_loss.detach().item()

        return total_policy_loss / self.STEP, total_value_loss / self.STEP
    
def train():
    num_states = train_env.observation_space.shape[0]
    num_actions = train_env.action_space.n
    hidden_size = 128
    discount = 0.99
    max_epochs = 2000
    clip_r = 0.2
    observation_per_epoch = 20
    STEP = 5
    N_TRIALS = 25
    train_rewards = []
    test_rewards = []

    agent = PEMO(num_states, num_actions, hidden_size, discount, clip_r, STEP)

    for iteration in range(max_epochs):
        
        agent.Network.train()

        train_reward, state, action, log_prob, returns, advantages = agent.get_training_data()
        
        policy_loss, value_loss = agent.update(state, action, log_prob, returns, advantages)
        
        agent.Network.eval()

        state, _ = test_env.reset()

        test_reward = 0

        ep = 0

        while True:
            with torch.no_grad():
                action_dist = agent.Network.Policy(torch.Tensor(state))
                action_prob = F.softmax(action_dist, dim = -1)

            action = torch.argmax(action_prob, dim = -1)

            state, reward, done, truncated, _ = test_env.step(action.detach().numpy())
            test_reward += reward

            ep += 1
            if done or truncated:
                break

        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train = np.mean(train_rewards[-N_TRIALS:])
        mean_test = np.mean(test_rewards[-N_TRIALS:])

        writer.add_scalar('Train/policy_loss', policy_loss, iteration)
        writer.add_scalar('Train/value_loss', value_loss, iteration)
        writer.add_scalar('Train/mean_reward', mean_train, iteration)
        writer.add_scalar('Train/reward', train_reward, iteration)
        writer.add_scalar('Test/reward', test_reward, iteration)
        writer.add_scalar('Test/mean_reward', mean_test, iteration)

        if iteration % observation_per_epoch == 0:   
            print(f'| Epsiodes: {iteration:4} | Mean Train: {mean_train: 7.1f} | Mean Test: {mean_test: 7.1f} | Length: {ep:5} | Test Reward: {test_reward: 7.1f}')

        if mean_test >= 200:
            break

    save_as_gif = False
    from matplotlib import animation
    import matplotlib.pyplot as plt
    def save_frames_as_gif(frames, path='./', filename='PPO.gif'):

        #Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=60)
        
    if save_as_gif:
        env_test = gym.make('LunarLander-v2', render_mode='rgb_array')
        frames = []
        state, _ = env_test.reset()
        state = torch.Tensor(state)
        episode_reward = 0
        while True:
            action = agent.Network.Policy(state)
            action = F.softmax(action, dim = -1)
            action = torch.argmax(action, dim = -1)

            next_state, reward, done, truncated, _ = env_test.step(action.detach().numpy())
            
            frames.append(env_test.render())
            episode_reward += reward
            next_state = torch.Tensor(next_state)
            state = next_state
            
            if done or truncated:
                break
        env_test.close()
        save_frames_as_gif(frames)

train_env = gym.make('LunarLander-v2')
test_env = gym.make('LunarLander-v2')

train()
print('Success!')