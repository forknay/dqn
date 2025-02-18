import random
import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

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
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        self.layer1 = torch.nn.Linear(self.nb_states, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, self.nb_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x) 
    
def action(state):
    if np.random.rand() <= epsilon:
        action = random.randrange(nb_actions)
        print("Random action:", action.item())
        return torch.tensor([[env.action_space.sample()]], device="cpu", dtype=torch.long)
    else:
        with torch.no_grad():
            action = policy_net(state).max(1).view(1, 1)
            print("Predicted action: ", action.item())
            return action

class memory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + discount * np.amax(self.model.predict(next_state, verbose=0)[0]) # Find the Q-value of the best next action
            target_f = self.model.predict(state, verbose=0) # Get the current Q-values
            #print("Reward:", reward, "Target:", target, "Target_f:", target_f)
            target_f[0][action] = target # Update the Q-value of the chosen next action

            self.model.fit(state, target_f, epochs=1, verbose=0) # Train the model on the updated Q-values

        dqn.epsilon = max(dqn.epsilon * dqn.epsilon_decay, dqn.min_epsilon)

    
nb_episodes = 1000
memory = []
capacity = 1000
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01 
discount = 0.99 # Discourage taking longer than needed, doesnt really matter for cartpole since no good "end"

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state, info = env.reset() 

    nb_states = len(state)
    nb_actions = env.action_space.n
    print(nb_states, nb_actions)

    policy_net = DQN(nb_states, nb_actions).to("cpu")
    target_net = DQN(nb_states, nb_actions).to("cpu")

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001 , amsgrad=True)
    mem = memory(capacity)

    batch_size = 32
    scores = []

    #fp = open("results.txt", "a")

    for e in range(nb_episodes):
        mem_index = 0
        state, info = env.reset()
        print("State:", state[0], (1, nb_states))
        state = np.reshape(state[0], (1, nb_states))
        print("----",state)
        done = False

        if e % 5 == 0:

                print("---\n" * 5)
                print(scores)
                print("---\n" * 5)

        for time in range(501): # Avoid unlimited cartpole, could use while not done for other environments
            action = dqn.action(state)
            next_state, reward, done, _, __ = env.step(action) # Execute step
            #print("---")
            #print("Reward:", reward, "Done:", done)
            #print("Next state:", next_state, (1, nb_states))
            #print("---")
            next_state = np.reshape(next_state, (1, nb_states))
            if done:
                reward = -10 # Penalize falling off the cart

            dqn.memory.insert(mem_index % dqn.capacity-1, (state, action, reward, next_state, done)) # Add step to the replay memory
            mem_index += 1

            state = next_state
            if len(mem) > batch_size:
                #print("Replay -")
                mem.replay(batch_size)
            
            if done or time == 500:
                print("---\n" * 5)
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, nb_episodes, time, dqn.epsilon))
                print(scores)
                print("---\n" * 5)
                scores.append(time)

                #fp.write(str(time) + "\n")
                break

            
