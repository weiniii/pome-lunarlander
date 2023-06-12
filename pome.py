import numpy as np
import torch
import torch.nn as nn
import gym
from torch.optim import Adam
from collections import namedtuple
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from scipy.signal import lfilter
from torch.utils.tensorboard import SummaryWriter

PATH = "KL"
writer = SummaryWriter(f"./tb_record_POME_{PATH}")

Transition = namedtuple(
    'Transition', ('state', 'action', 'log_prob', 'next_state', 'reward'))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.Net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            # nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_dim),
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = nn.init.xavier_normal_(m.weight.data)

    def forward(self, inputs):
        outputs = self.Net(inputs)
        return outputs
    
'''
Four Netowrks - Critic(Value), Policy(Actor), State(transtition), Reward
Actor Critic for model-free
State Reward for model-based
'''
class Network(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size):
        super().__init__()
        self.Critic = MLP(num_states, hidden_size, 1)
        self.Policy = MLP(num_states, hidden_size, num_actions)
        self.State = MLP(num_states + 1, hidden_size, num_states)
        self.Reward = MLP(num_states + 1, hidden_size, 1)

'''
POME implementation - 

Introduction for variables - 

    NUM_STATES - State dimension
    NUM_ATIONS - Action dimension
    HIDDEN_SIZE - Hidden layers size
    DISCOUNT - Discount factor 
    ALPHA - Clip ratio for epsilon
    CLIP_R - Clip ratio for A^hat
    CV - Value loss ratio 
    CT - Transition loss ratio
    POME_STEPS - The number of updating times per sample
    BETA - Ratio for KL divergence
    POME_LR - Policy learning
    REWARD_LR - reward learning
    Normalize - if normalize the returns and A^hat
'''
class PEMO(object):
    def __init__(self, NUM_STATES, NUM_ATIONS, HIDDEN_SIZE, DISCOUNT, ALPHA, CLIP_R, CV, CT, POME_STEPS, BETA, POME_LR, REWARD_LR):
        self.num_actions = NUM_ATIONS
        self.num_states = NUM_STATES
        self.hidden_size = HIDDEN_SIZE
        self.discount = DISCOUNT
        self.alpha = ALPHA
        self.clip_r = CLIP_R
        self.cv = CV
        self.ct = CT
        self.Normalize = True
        self.POME_steps = POME_STEPS
        self.beta = BETA
        self.Network = Network(self.num_states, self.num_actions, self.hidden_size)
        self.Network_optim = Adam([{'params': self.Network.Critic.parameters()},
                                   {'params': self.Network.Policy.parameters()},
                                   {'params': self.Network.State.parameters()}], lr=POME_LR)

        self.Reward = MLP(self.num_states + 1, self.hidden_size, 1)
        self.Reward_optim = Adam(self.Network.Reward.parameters(), lr=REWARD_LR)


    '''
    Sample trajectory
    '''
    def get_info_from_env(self):
        
        Trajectory = []

        state, _ = env.reset()

        with torch.no_grad():
            state = torch.Tensor(state)
            train_reward = 0
            
            while True:
                
                action_dist = self.Network.Policy(state)

                action_prob = F.softmax(action_dist, dim = -1)

                action_categorical = Categorical(action_prob)

                action = action_categorical.sample()

                log_prob = action_categorical.log_prob(action)

                next_state, reward, done, truncated, _ = env.step(action.detach().numpy())
                
                train_reward += reward

                next_state = torch.Tensor(next_state)

                reward = torch.Tensor([reward])
                
                Trajectory.append(Transition(state, action, log_prob, next_state, reward))

                state = next_state

                if done or truncated:
                    break

        return train_reward, Transition(*zip(*Trajectory))
    
    '''
    Get training data from sampling
    '''
    def get_training_data(self):

        train_reward, trajectory = self.get_info_from_env()
            
        reward = torch.Tensor(trajectory.reward).unsqueeze(1)
        state = torch.stack(list(trajectory.state), dim=0)
        next_state = torch.stack(list(trajectory.next_state), dim=0)
        log_prob = torch.Tensor(trajectory.log_prob).unsqueeze(1)
        action = torch.Tensor(trajectory.action).unsqueeze(1)

        returns = self.Cumulation(reward, self.discount).unsqueeze(1)
    
        return train_reward, reward, state, action, next_state, log_prob, returns
    
    '''
    Cumulation function for returns and A^hat
    '''
    def Cumulation(self, inputs, factor):
        outputs = torch.Tensor(lfilter([1], [1, - factor], list(inputs.detach().squeeze(1).numpy())[::-1])[::-1].copy())
        
        if self.Normalize:
            return (outputs - torch.mean(outputs)) / torch.std(outputs)
        else:
            return outputs
    
    '''
    Update Four Networks. Policy, Critic, and State are updated together. Reward is updated itself.

    Introduction for variable - 

        Returns - returns reward. In gerenal, it is G(t)
        value_free - V(s_t) from the critic network
        bext_value_free - V(s_{t+1}) from the critic netwrork
        q_free - Q_{f,t}
        next_state_based - T^(s_t,a_t) from the state network
        reward_based - R^(s_t, a_t) from the reward network
        q_based - Q_{b,t}
        epsilon - epsilon in this paper
        mean_epsilon - the median of epsilons
        delta - delta in this paper
        q_pome - Q_POME in this paper
        delta_pome - delta_pome in this paper
        A_hat_pome - A_hat_pome in this paper
    '''
    def Update(self, state, next_state, action, returns, log_prob, reward, iteration):
        
        total_policy_loss = 0
        total_value_loss = 0
        total_state_loss = 0
        total_reward_loss = 0

        # reward_based = reward

        with torch.no_grad():
            value_free = self.Network.Critic(state)

            next_value_free = self.Network.Critic(next_state)
            q_free = reward + self.discount * next_value_free

            next_state_based = self.Network.State(torch.cat((state, action), dim=1))
            reward_based = self.Network.Reward(torch.cat((state, action), dim=1))

            next_value_based =  self.Network.Critic(next_state_based)
            q_based = reward_based + self.discount * next_value_based

            epsilon = torch.abs(q_free - q_based)
            mean_epsilon = torch.median(epsilon)

            delta = torch.abs(q_free - value_free)

            q_pome = q_free + self.alpha * torch.clamp(epsilon - mean_epsilon, -delta, delta)
                        
            delta_pome = q_pome - value_free
            
            A_hat_pome = self.Cumulation(delta_pome, self.discount).unsqueeze(1)


            # A_hat_pome = returns - value_free

        for _ in range(self.POME_steps):

            value_free = self.Network.Critic(state)

            next_state_based = self.Network.State(torch.cat((state, action), dim=1))
                        
            ''' 
            ratio 
            '''
            new_action_dist = self.Network.Policy(state)

            new_action_prob = F.softmax(new_action_dist, dim = -1)

            new_action_categorical = Categorical(new_action_prob)

            new_log_prob = new_action_categorical.log_prob(action.squeeze(1)).unsqueeze(1)
            
            ratio = torch.exp(new_log_prob - log_prob)
            '''
            Policy update
            '''
            policy_loss_1 = ratio * A_hat_pome

            policy_loss_2 = torch.clamp(ratio, 1 - self.clip_r, 1 + self.clip_r) * A_hat_pome
            
            l_pome = torch.min(policy_loss_1, policy_loss_2)
            
            kl_diverge = torch.exp(log_prob) * (log_prob - new_log_prob) * self.beta

            pome_loss = - torch.mean(l_pome - kl_diverge)

            '''
            State update
            '''

            state_loss = F.smooth_l1_loss(next_state, next_state_based)

            '''
            Value update
            '''

            value_loss = F.mse_loss(returns + A_hat_pome, value_free)

            '''
            Update together
            '''

            total_loss = pome_loss + value_loss * self.cv + state_loss * self.ct

            self.Network_optim.zero_grad()
            
            total_loss.backward()
            
            self.Network_optim.step()

            total_policy_loss += pome_loss.detach().item()
            total_value_loss += value_loss.detach().item()
            total_state_loss += state_loss.detach().item()

        '''
        Reward update, you can remove reward network, i.e., let reward_based = reward
        '''
        for __ in range(self.POME_steps):
            
            reward_based = self.Network.Reward(torch.cat((state, action), dim=1))

            reward_loss = F.mse_loss(reward, reward_based)
            

            self.Reward_optim.zero_grad()
            
            reward_loss.backward()
            
            self.Reward_optim.step()

            total_reward_loss += reward_loss.detach().item()

        return total_policy_loss / self.POME_steps, total_value_loss / self.POME_steps, total_state_loss / self.POME_steps, total_reward_loss / self.POME_steps

def train():
    NUM_STATES = env.observation_space.shape[0]
    NUM_ATIONS = env.action_space.n
    HIDDEN_SIZE = 128
    DISCOUNT = 0.99
    MAT_ITERATION = 1000
    CLIP_RATOPN = 0.2
    ALPHA = 0.1
    OBSERVATION_PER_EPOCH = 20
    CV = 1
    CT = 2
    N_TRIAL = 20
    POME_STEPS = 5
    BETA = 0.01
    POME_LR = 5e-3
    REWARD_LR = 1e-2
    train_rewards = []
    test_rewards = []
    average_reward = 0
    agent = PEMO(NUM_STATES, NUM_ATIONS, HIDDEN_SIZE, DISCOUNT, ALPHA, CLIP_RATOPN, CV, CT, POME_STEPS, BETA, POME_LR, REWARD_LR)

    scheduler_pome = Scheduler.StepLR(agent.Network_optim, step_size=100, gamma=0.9)
    scheduler_reward = Scheduler.StepLR(agent.Reward_optim, step_size=100, gamma=0.9)

    for iteration in range(1, MAT_ITERATION + 1):
        
        agent.Network.train()
        

        train_reward, reward, state, action, next_state, log_prob, returns = agent.get_training_data()

        
        policy_loss, value_loss, state_loss, reward_loss = agent.Update(state, next_state, action, returns, log_prob, reward, iteration)
        
        scheduler_pome.step()
        scheduler_reward.step()

        agent.Network.eval()
        state, _ = env.reset()
        test_reward = 0
        ep = 0

        while True:
            action_dist = agent.Network.Policy(torch.Tensor(state))
            action_prob = F.softmax(action_dist, dim = -1)
            action = torch.argmax(action_prob, dim = -1)
            next_state, reward, done, truncated, _ = env.step(action.detach().numpy())
            test_reward += reward
            state = next_state
            ep += 1
            if done or truncated:
                break
        
        train_rewards.append(train_reward)
        test_rewards.append(test_reward)
        mean_train = np.mean(train_rewards[-N_TRIAL:])
        mean_test = np.mean(test_rewards[-N_TRIAL:])

        average_reward = 0.05 * test_reward + 0.95 * average_reward

        writer.add_scalar('Train/policy_loss', policy_loss, iteration)
        writer.add_scalar('Train/value_loss', value_loss, iteration)
        writer.add_scalar('Train/state_loss', state_loss, iteration)
        writer.add_scalar('Train/reward_loss', reward_loss, iteration)
        writer.add_scalar('Train/mean_reward', mean_train, iteration)
        writer.add_scalar('Train/reward', train_reward, iteration)
        writer.add_scalar('Test/reward', test_reward, iteration)
        writer.add_scalar('Test/mean_reward', mean_test, iteration)

        if iteration % OBSERVATION_PER_EPOCH == 0:

            print(f'| Epsiodes: {iteration:4} | Mean Train: {mean_train: 7.1f} | Mean Test: {mean_test: 7.1f} | Length: {ep:5} | Test Reward: {test_reward: 7.1f}')

        if mean_test >= 200:
            break

    save_as_gif = False
    from matplotlib import animation
    import matplotlib.pyplot as plt
    def save_frames_as_gif(frames, path="./", filename=f'{PATH}.gif'):

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



env = gym.make('LunarLander-v2')
train()
print('Done!')