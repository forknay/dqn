import random
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque

plt.ion()

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
    action = env.action_space.sample()  # ()
    print("Random action: ", action)
    if torch.rand(1).item() <= epsilon:
        return action
    else:
        with torch.no_grad():
            action = policy_net(state).max(0)[1] # ()
            print("Predicted action: ", action)
            return action.item()
        
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) 

    def __len__(self):
        return len(self.memory)
    
    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size) # (batch_size, (state, action, next_state, reward))
        states, actions, next_states, rewards = zip(*batch) # Separate the diff components into lists (for each transition)
        #Compute all non final states
        non_final_mask = torch.tensor([s is not None for s in next_states], dtype=torch.bool)  # (batch_size,)
        non_final_next_states = torch.stack([s for s in next_states if s is not None]) # (N, nb_states)
        # Return if no non-final states
        if non_final_next_states.shape[0] == 0:
            return
        
        # Compute all replays at once rather than one by one in a for loop
        state_batch = torch.stack(states)
        action_batch = torch.stack(actions)
        reward_batch = torch.tensor(rewards)
        
        # Gather values for actions taken over dimension 1 (action dim), for each action in action_batch
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Target net max action
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            # Returns max value for each non-final state
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * DISCOUNT) + reward_batch # That one Q formula 

       #Compute loss for graph
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1)) # (batch_size, 1) -> ()

        optimizer.zero_grad() # Clear gradients
        loss.backward()
        # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state, info = env.reset() 

    nb_states = len(state) # Number of input variables (4 for CartPole)
    nb_actions = env.action_space.n # Number of possible actions (2 for CartPole)

    # Build policy and target networks
    policy_net = DQN(nb_states, nb_actions)
    target_net = DQN(nb_states, nb_actions)
    # Sync target net with policy net
    target_net.load_state_dict(policy_net.state_dict())
    # Initialize target net optimizer (no need for target since it "follows" policy)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR , amsgrad=True)

    mem = ReplayMemory(CAPACITY)
    mem_index = 0
    epsilon = EPS
    episode_durations = []
    scores = []

    # Training loop
    for ep in range(N_EPISODES):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32) # (nb_states,)

        for time in range(501): # Avoid unlimited cartpole, could use while not done for other environments
            action_taken = action(state, epsilon) # (1,)
            print("Action taken: ", action_taken)
            next_state, reward, done, _, __ = env.step(action_taken) # Execute step

            if done:
                next_state = None
                reward = -10
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32) # (nb_states,)

            # Store transition in memory
            mem.memory.append((state, torch.tensor([action_taken], dtype=torch.long), next_state, float(reward)))
            mem_index += 1
            state = next_state

            if len(mem) >= BATCH_SIZE:
                mem.replay(BATCH_SIZE)
                
            # Soft-update/follow target net
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in target_net_state_dict:
                 # Keeps mostly target weights intact, but slowly follows policy net
                 target_net_state_dict[key] = policy_net_state_dict[key]* TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)
            # Break at the end of episode or if time limit reached
            if done or time == 500:
                episode_durations.append(time + 1)
                plot_durations()
                scores.append(time)
                break
        # Decay epsilon every episode
        epsilon = max(epsilon * EPS_DECAY, MIN_EPS)

print("Done")
plot_durations(show_result=True)
plt.ioff()
plt.show()

            
