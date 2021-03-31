from tensorforce.execution import Runner
from tensorforce import Agent, Environment

environment = Environment.create(environment='gym', level='CartPole', max_episode_timesteps=500)

flag_restart_agent = False
flag_evaluate = True

if flag_restart_agent:
    agent = Agent.create(
        agent='tensorforce', 
        environment=environment, 
        update=64,
        optimizer=dict(optimizer='adam', learning_rate=1e-3),
        objective='policy_gradient', reward_estimation=dict(horizon=20),
        memory=dict(capacity=10000),
        summarizer=dict(
            directory='data/cartpole/summaries',
            # list of labels, or 'all'
            # labels='all',
            # ['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
        ),
        saver=dict(
            directory='data/cartpole/checkpoints',
            frequency=100  # save checkpoint every 100 updates
        ),
        
    )
else:
    agent = Agent.load(directory='data/cartpole/checkpoints')

if not flag_evaluate:
    runner = Runner(
        agent=agent,
        environment=environment,
        # num_parallel=5, remote='multiprocessing',
    )
    runner.run(num_episodes=200, evaluation=True,
    )
    runner.close()
else:
    environment.visualize=True
    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for _ in range(100):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals,
                independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward

    print('Mean episode reward:', sum_rewards / 100)

    # Close agent and environment
    agent.close()
    environment.close()

# Check performances on Tensorboard
# !pip show tensorboard
# !tensorboard --logdir data/summaries