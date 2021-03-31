### Dynamic Plot Example
import numpy as np
import time
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.cos(x)

plt.ion()

fig, ax = plt.subplots(figsize=(8,6))
line1, = ax.plot(0, 0)
plt.title("Dynamic Plot of sinx",fontsize=25)

plt.xlabel("X",fontsize=18)
plt.ylabel("sinX",fontsize=18)

for p in range(100):
    updated_y = np.cos(x-0.05*p)
    
    ### USE SAME LINE OBJECT (need to adapt zoom)
    # line1.set_xdata(x), line1.set_ydata(updated_y)
    # ax.relim(), ax.autoscale_view()

    ### CREATE NEW LINE EACH TIME (Need to clear previous one before)
    if not plt.fignum_exists(fig.number):
        fig, ax = plt.subplots()
    else:
        ax.clear()
    ax.plot(x, updated_y)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.001)