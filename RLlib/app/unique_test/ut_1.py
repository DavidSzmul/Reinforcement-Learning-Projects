import os, sys
sys.path.append(os.getcwd()) #### TODEBUG
import numpy as np
import matplotlib.pyplot as plt
from library.Agent import Displayer
ep_rewards = np.load('unique_test/historic_reward_DQN.npy')

### DISPLAY
# ep_rewards = np.random.rand(100)
t = range(len(ep_rewards))
Nb_mean = 10
rewards_stat = np.zeros((3, len(ep_rewards)))#Min, Mean, Max
for i in range(Nb_mean, len(ep_rewards)):
    rewards_stat[0,i] = min(ep_rewards[i-Nb_mean:i+1])
    rewards_stat[1,i] = sum(ep_rewards[i-Nb_mean:i+1])/len(ep_rewards[i-Nb_mean:i+1])
    rewards_stat[2,i] = max(ep_rewards[i-Nb_mean:i+1])

mysignals = [
    {'name': 'Min Reward', 
    'x': t, 'y': rewards_stat[0,:],
    'linewidth':1},
    {'name': 'Mean Reward',
    'x': t, 'y': rewards_stat[1,:],
    'linewidth':1},
    {'name': 'Max Reward',
    'x': t, 'y': rewards_stat[2,:],
    'linewidth':1}
    ]

fig, ax = plt.subplots()
for signal in mysignals:
    ax.plot(signal['x'], signal['y'], 
            # color=signal['color'], 
            linewidth=signal['linewidth'],
            label=signal['name'])
# Enable legend
ax.legend()
# ax.set_title("My graph")
plt.show()

