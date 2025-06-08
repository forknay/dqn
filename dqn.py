import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import namedtuple

plt.ion()

episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
# Hyperparameters
N_EPISODES = 500
CAPACITY = 10000
MIN_EPS = 0.01
DISCOUNT = 0.99
LR = 0.001
BATCH_SIZE = 32
# Epsilon-greedy
EPS = 1.0
EPS_DECAY = 0.995
# Double DQN
TAU = 0.005
# Noisy Nets
SIGMA = 0.017
class NoisyLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma = SIGMA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.mu_bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features))


class DQN(torch.nn.Module):
    def __init__(self, nb_states, nb_actions):
        super().__init__()
        self.layer1 = torch.nn.Linear(nb_states, 24)
        self.layer2 = torch.nn.Linear(24, 24)
        self.layer3 = torch.nn.Linear(24, nb_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) 
    
def action(state, epsilon):
    action = env.action_space.sample()  # Random action
    print("Random action: ", action)
    if np.random.rand() <= epsilon:
        return torch.tensor([[action]])
    else:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
            print("Predicted action: ", action.squeeze(0).item())
            return action
        
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*batch)) #Idk man, this is just magic (unzip)
        #Compute all non final states
        non_final_mask = torch.tensor([s is not None for s in batch.next_state])
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Compute all replays at once rather than one by one in a for loop
        state_batch = torch.cat(batch.state)
        #print("Action batch:", batch.action)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Policy net preferred action
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Target net max action
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * DISCOUNT) + reward_batch

       #Compute loss for graph
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state, info = env.reset() 

    nb_states = len(state)
    nb_actions = env.action_space.n
    policy_net = DQN(nb_states, nb_actions)
    target_net = DQN(nb_states, nb_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR , amsgrad=True)
    mem = memory(CAPACITY)

    scores = []
    epsilon = EPS
    mem_index = 0

    for ep in range(N_EPISODES):
        state, info = env.reset()
        #print("State:", state[0], (1, nb_states))
        #print("----",state)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False

        #if e % 5 == 0:

                #print("---\n" * 5)
                #print(scores)
                #print("---\n" * 5)

        for time in range(501): # Avoid unlimited cartpole, could use while not done for other environments
            action_taken = action(state, epsilon)
            next_state, reward, done, _, __ = env.step(action_taken.item()) # Execute step
            #print("---")
            #print("Reward:", reward, "Done:", done)
            #print("Next state:", next_state, (1, nb_states))
            #print("---")
             # Penalize falling off the cart
            
            if done:
                reward = -10
            
            reward = torch.tensor([reward], dtype=torch.float32)
            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            
            mem.memory.insert(mem_index % (mem.capacity-1), (state, action_taken, next_state, reward))
            mem_index += 1

            state = next_state
            if len(mem.memory) > BATCH_SIZE:
                #print("Replay -")
                mem.replay(BATCH_SIZE)
                epsilon = max(epsilon * EPS_DECAY, MIN_EPS)

            # Update target net
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in target_net_state_dict:
                 target_net_state_dict[key] = policy_net_state_dict[key]* TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            
            if done or time == 500:
                episode_durations.append(time + 1)
                plot_durations()
                print("---\n" * 5)
                print("episode: {}/{}, score: {}, e: {:.2}".format(ep, N_EPISODES, time, epsilon))
                print(scores)
                print("---\n" * 5)
                scores.append(time)

                #fp.write(str(time) + "\n")
                break
print("Done")
plot_durations(show_result=True)
plt.ioff()
plt.show()

            
