import numpy as np
from collections import deque
import random
from tqdm import tqdm
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam


class DQN_Agent:
    def __init__(self, env, 
    automatic_model=True, layers_model = [32, 32], # In case of auto-generated model
    loading_model=False, name_model='', model=None,   # In case of loaded model (or model directly)
    gamma=0.95, epsilon_min = 0.01, epsilon_decay = 0.995, learning_rate=1e-3,
    memory_size=50000, batch_size=64, nb_update_target=100):

        # DQN Parameters
        self.gamma = gamma  # discount rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        # Global Parameters
        self.env= env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.nb_update_target = nb_update_target

        # Variables
        self.epsilon = 1.0  # exploration rate (variable)
        self.target_update_counter = 0 # Used to count when to update target network with main network's weights

        # Model for DQN
        if loading_model: # Load Existing Model
            self.model = load_model(name_model) 
            print('DQN Model loaded')
        elif automatic_model: # Generate Automatic Model
            self.model = self._build_model(layers_model)
            print('DQN Model builded')
        else: # Or directly in input
            assert model is not None, 'No model set in input'
            self.model = model

        # Generate target
        self.target_model=tf.keras.models.clone_model(self.model)
        self.target_model.set_weights(self.model.get_weights()) 

        
    def _build_model(self, layers_model):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(layers_model[0], activation='relu', input_shape=env.observation_space.shape))
        model.add(BatchNormalization())
        for layer in layers_model[1:]:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization())
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss="mse", 
                      optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def memorize(self, transition):
        self.memory.append(transition)

    # Trains main network every step during episode
    def train(self):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < self.batch_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.memory, self.batch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        # Now we need to enumerate our batches
        state_batch, Q_value_batch = [], []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            # And append to our training data
            state_batch.append(current_state)
            Q_value_batch.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(state_batch), np.array(Q_value_batch), batch_size=self.batch_size, verbose=0, shuffle=False,)
        
        # If counter reaches set value, update target network with weights of main network
        self.target_update_counter +=1
        if self.target_update_counter >= self.nb_update_target:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            print('Target updated')

        # Update Exploration epsilon number
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min) 

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, len(state)))[0]

    def get_action(self, state):
        # Get best action from Q depending on model
        return np.argmax(self.get_qs(state))

    def get_action_training(self, state):
        # Exploration during training
        if np.random.random() > self.epsilon:
            return self.get_action(state)
        else:
            # Get random action
            return self.env.action_space.sample()

if __name__ == '__main__':
    import os, sys, time
    import matplotlib.pyplot as plt

    ### Use of GPU ?
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    flag_use_GPU = True
    os.environ["CUDA_VISIBLE_DEVICES"]=["-1", "0"][flag_use_GPU]
    sys.path.append(os.getcwd())

    ### LIBRARIES
    from library.Environment import Environment
    from library.Network import NetworkGenerator
    from library.Agent import Displayer

    # Environment settings
    MODEL_NAME = 'CartePole'    

    ### TRAINING/EXPLORATION
    EPISODES = 2000
    # EPSILON_DECAY = pow(MIN_EPSILON, 1/(NB_MAX_EXPLORATION)) 

    # Display Results
    SHOW_PREVIEW = False
    DELTA_SAVE = 500  # episodes
    verbose = 1

    ### INITIALIZATION
    env = Environment("gym", 'CartPole-v0')
    env = env.getEnv()
    Disp = Displayer(delta_display=10) # Used to display performances
    
    flag_load_model = False
    path_best_Model = 'models/.model'

    if flag_load_model:
        agent = DQN_Agent(env, loading_model=True, name_model=path_best_Model)
    else:
        agent = DQN_Agent(env, layers_model=[32,32])
    
    # LOOP Episode
    ep_rewards=[]
    # for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    for episode in range(1, EPISODES + 1):
        # Restarting episode - Environment
        episode_reward = 0
        state = env.reset()
        done = False
        
        # LOOP Step
        while not done:       

            # Determine action to do     
            action = agent.get_action_training(state)
            new_state, reward, done, _ = env.step(action)
            transition = (state, action, reward, new_state, done)
            episode_reward += reward
            state = new_state

            if SHOW_PREVIEW:
                env.render()
                
            # Every step we update replay memory and train main network
            agent.memorize(transition)
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(episode, EPISODES, episode_reward))
        # END Episode   
        agent.train()
        ep_rewards.append(episode_reward)    

        # SAVE MODEL
        if not episode % DELTA_SAVE: # and min_reward >= MIN_REWARD:
            # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
            avg_reward = sum(ep_rewards[-DELTA_SAVE:])/len(ep_rewards[-DELTA_SAVE:])
            min_reward = min(ep_rewards[-DELTA_SAVE:])
            max_reward = max(ep_rewards[-DELTA_SAVE:])
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        
        # DISPLAY HISTORIC
        if verbose==1:
            Disp.display_historic(agent.epsilon, episode_reward) 
    
    ### END OF EPISODES/TRAINING
    env.close()
    np.save('historic_reward', ep_rewards)

    # END OF DISPLAY
    plt.ioff(), plt.show()
    
