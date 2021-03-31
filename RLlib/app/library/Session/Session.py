import os, sys
sys.path.append(os.getcwd()) #### TODEBUG
from library.Agent import Displayer
class Session(object):

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, nb_episode=1, delta_save=None, verbose=0, name_model = 'test'):
        Disp = Displayer(delta_display=10) # Used to display performances

        # LOOP Episode
        ep_rewards=[]
        # for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        for episode in range(1, nb_episode + 1):
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
                agent.memorize(transition)
                if done:
                    # print the score and break out of the loop
                    print("episode: {}/{}, score: {}"
                        .format(episode, nb_episode, episode_reward))
            # END Episode   
            agent.train()
            ep_rewards.append(episode_reward)    

            # SAVE MODEL
            if delta_save is not None and not episode % delta_save: # and min_reward >= MIN_REWARD:
                # agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
                avg_reward = sum(ep_rewards[-delta_save:])/len(ep_rewards[-delta_save:])
                min_reward = min(ep_rewards[-delta_save:])
                max_reward = max(ep_rewards[-delta_save:])
                agent.model.save(f'models/{name_model}__{max_reward:_>7.2f}max_{avg_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            
            # DISPLAY HISTORIC
            if verbose==1:
                Disp.display_historic(agent.epsilon, episode_reward) 
        
        ### END OF EPISODES/TRAINING
        env.close()
        np.save('historic_reward', ep_rewards)

        # END OF DISPLAY
        plt.ioff(), plt.show()

    def test(self, nb_test=1):

        for _ in range(1, nb_test + 1):
            # Restarting episode - Environment
            state = self.env.reset()
            done = False
            
            # LOOP Step
            while not done:       
                # Determine action to do     
                action = self.agent.get_action(state)
                new_state, _, done, _ = self.env.step(action)
                state = new_state
                self.env.render()
            env.close()

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
    from library.Agent import DQN_Agent, Displayer

    ### INITIALIZATION
    env = Environment("gym", 'CartPole-v0')
    env = env.getEnv()    
    path_best_Model = 'models\Best_Models\DQN_Cartepole_BestModel.model'
    agent = DQN_Agent(env, loading_model=True, name_model=path_best_Model)
    session = Session(env, agent)

    ### TRAIN
    session.train(nb_episode=200, verbose=1, name_model='CartePole')
    ### TEST
    # session.test(nb_test=2)