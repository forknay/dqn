import random
import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

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
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



class DQN(torch.nn.Module):
    def __init__(self, nb_states, nb_actions):
        super(DQN, self).__init__()
        
        self.layer1 = torch.nn.Linear(nb_states, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, nb_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x) 
    
def action(state):
    if np.random.rand() <= epsilon:
        action = random.randrange(nb_actions)
        print("Random action:", action)
        return torch.tensor([[env.action_space.sample()]], device="cpu", dtype=torch.long)
    else:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].view(1, 1)
            print("Predicted action: ", action)
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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device="cpu", dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        # Compute all replays at once rather than one by one in a for loop
        state_batch = torch.cat(batch.state)
        #print("Action batch:", batch.action)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        # Policy net preferred action
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Target net max action
        next_state_values = torch.zeros(batch_size, device="cpu")
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
        expected_state_action_values = (next_state_values * discount) + reward_batch

       #Compute loss for graph
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

        
        

    
nb_episodes = 1000
capacity = 1000
epsilon = 1.0
epsilon_decay = 0.9995
min_epsilon = 0.01 
discount = 0.99 # Discourage taking longer than needed, doesnt really matter for cartpole since no good "end"
tau = 0.005

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state, info = env.reset() 

    nb_states = len(state)
    nb_actions = env.action_space.n
    print(nb_states, nb_actions)

    policy_net = DQN(nb_states, nb_actions).to("cpu")
    target_net = DQN(nb_states, nb_actions).to("cpu")
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001 , amsgrad=True)
    mem = memory(capacity)

    batch_size = 32
    scores = []

    #fp = open("results.txt", "a")

    for e in range(nb_episodes):
        mem_index = 0
        state, info = env.reset()
        print("State:", state[0], (1, nb_states))
        print("----",state)
        state = torch.tensor(state, dtype=torch.float32, device="cpu").unsqueeze(0)
        done = False

        if e % 5 == 0:

                print("---\n" * 5)
                print(scores)
                print("---\n" * 5)

        for time in range(501): # Avoid unlimited cartpole, could use while not done for other environments
            action_taken = action(state)
            next_state, reward, done, _, __ = env.step(action_taken.item()) # Execute step
            #print("---")
            #print("Reward:", reward, "Done:", done)
            #print("Next state:", next_state, (1, nb_states))
            #print("---")
             # Penalize falling off the cart
            

            reward = torch.tensor([reward], device="cpu")
            if done:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32, device="cpu").unsqueeze(0)
            
            mem.memory.insert(mem_index % (mem.capacity-1), (state, action_taken, next_state, reward))
            mem_index += 1

            state = next_state
            if len(mem.memory) > batch_size:
                #print("Replay -")
                mem.replay(batch_size)
                epsilon = max(epsilon * epsilon_decay, min_epsilon)

            # Update target net
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in target_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]* tau + target_net_state_dict[key] * (1 - tau)
            target_net.load_state_dict(target_net_state_dict)
            
            if done or time == 500:
                episode_durations.append(time + 1)
                plot_durations()
                print("---\n" * 5)
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, nb_episodes, time, epsilon))
                print(scores)
                print("---\n" * 5)
                scores.append(time)

                #fp.write(str(time) + "\n")
                break
print("Done")
plot_durations(show_result=True)
plt.ioff()
plt.show()

            
