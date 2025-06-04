import random
import gym
import numpy as np
import keras
from tensorflow.dqn import DQN

nb_episodes = 1000

env = gym.make('CartPole-v1')
nb_states = env.observation_space.shape[0]
nb_actions = env.action_space.n
print(nb_states, nb_actions)
dqn = DQN(nb_states, nb_actions)
dqn.load('cartpole.weights.h5')
dqn.epsilon = 0.0
scores = []

fp = open("results.txt", "a")

for e in range(nb_episodes):
    state = env.reset()
    state = np.reshape(state[0], (1, nb_states))
    done = False

    for time in range(501): # Avoid unlimited cartpole, could use while not done for other environments
        action = dqn.action(state)
        next_state, reward, done, _, __ = env.step(action) # Execute step
        #print("---")
        #print("Reward:", reward, "Done:", done)
        #print("Next state:", next_state, (1, nb_states))
        #print("---")
        next_state = np.reshape(next_state, (1, nb_states))

        state = next_state

        if done or time == 500:
            print("---\n" * 5)
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, nb_episodes, time, dqn.epsilon))
            print(scores)
            print("---\n" * 5)
            scores.append(time)
            fp.write(str(time) + "\n")
            break