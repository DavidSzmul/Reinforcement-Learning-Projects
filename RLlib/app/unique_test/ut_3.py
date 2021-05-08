from library.Environment import Environment
from library.Agent import DQN_Agent

# Environment settings
MODEL_NAME = 'CartePole'    


### INITIALIZATION
env = Environment("gym", 'CartPole-v0')
env = env.getEnv()

USE_SOFT_UPDATE=False
USE_DOUBLE_DQN=True
USE_PER=True
verbose = 1

flag_load_model = True
path_best_Model = 'models/Best_Models/best_model.model'
if flag_load_model:
    agent = DQN_Agent(env, loading_model=True, use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER, name_model=path_best_Model)
else:
    agent = DQN_Agent(env, layers_model=[24,24],use_soft_update=USE_SOFT_UPDATE, use_double_dqn=USE_DOUBLE_DQN, use_PER=USE_PER)

### TRAINING/EXPLORATION
STEPS_MAX = 1e4
DELTA_SAVE = 0 
# agent.train(nb_steps=STEPS_MAX,delta_save=DELTA_SAVE, verbose=verbose, name_model = path_best_Model)
agent.test(nb_test=2)
