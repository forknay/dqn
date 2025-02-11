import random
import gym
import numpy as np
import keras

class DQN():
    def __init__(self, nb_states, nb_actions):
        self.memory = []
        self.capacity = 1000
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.model = self._build_model()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01 
        self.discount = 0.99 # Discourage taking longer than needed, doesnt really matter for cartpole since no good "end"

    def _build_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.nb_states, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.nb_actions, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.001))

        return model
    
    def action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.nb_actions)
            print("Random action:", action)
            return action
        else:
            q_values = self.model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            print("Predicted action: ", action)
            return action
        
    def replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.discount * np.amax(self.model.predict(next_state, verbose=0)[0]) # Find the Q-value of the best next action
            target_f = self.model.predict(state, verbose=0) # Get the current Q-values
            #print("Reward:", reward, "Target:", target, "Target_f:", target_f)
            target_f[0][action] = target # Update the Q-value of the chosen next action

            self.model.fit(state, target_f, epochs=1, verbose=0) # Train the model on the updated Q-values

        dqn.epsilon = max(dqn.epsilon * dqn.epsilon_decay, dqn.min_epsilon)

    def load(self, name):
        self.model.load_weights(name)
nb_episodes = 1000

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    nb_states = env.observation_space.shape[0]
    nb_actions = env.action_space.n
    print(nb_states, nb_actions)
    dqn = DQN(nb_states, nb_actions)
    dqn.load('cartpole.weights.h5')
    batch_size = 32
    scores = []

    #fp = open("results.txt", "a")

    for e in range(nb_episodes):
        mem_index = 0
        state = env.reset()
        print("State:", state[0], (1, nb_states))
        state = np.reshape(state[0], (1, nb_states))
        print("----",state)
        done = False

        if e % 5 == 0:
                dqn.model.save_weights('cartpole.weights.h5')
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
            if len(dqn.memory) > batch_size:
                #print("Replay -")
                dqn.replay(batch_size)
            
            if done or time == 500:
                print("---\n" * 5)
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, nb_episodes, time, dqn.epsilon))
                print(scores)
                print("---\n" * 5)
                scores.append(time)

                #fp.write(str(time) + "\n")
                break

            
