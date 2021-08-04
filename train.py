import gym
import numpy as np
import tensorflow as tf
import datetime
from ppo_agent import Agent 
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')

    # get action and state dimension
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]

    # run number in case you want to save logs from previous runs
    run_n = 5               # change this to change directories fro logs

    # PPO Agent variables
    update_every = 2000
    minibatch_size = 20
    n_epochs = 40
    alpha = 0.0003  # learning_rate
    gamma = 0.99
    lam = 0.95
    save_path = '/home/rwl/Desktop/Lunar_Lander_PPO' # adjust to your workspace
    
    # create TensorBoard writer
    test_log_dir = save_path + '/logs/' + str(run_n)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    average_reward = tf.keras.metrics.Mean('average_reward', dtype=tf.float32)

    # create an agent
    agent = Agent(state_size, action_size, 0.2, 0.5, minibatch_size, update_every, n_epochs, gamma, lam, alpha, save_path, test_summary_writer)

    # array to store episodic rewards
    score_history = []
 
    learn_iters = 0 # learning iterations peformed 
    update_steps = 0 # timesteps counter
    num_steps = 1000 # max number of episode steps
    n_games = 3000 # max number of episodes 
    print_freq = 5000
    save_after = 3000
    log_every_x_episodes = 5 
    print_running_reward = 0
    print_running_episodes = 0
    save_step = 0

    is_training = True

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
	episode_step = 0
        for t in range(1, num_steps+1):
            dist, val = agent.act(observation, is_training)
            action = dist.sample()
	    # print(action)
            observation_, reward, done, _ = env.step(action.numpy())

	    if done:
		mask = 0.0
	    else:
		mask = 1.0
            update_steps += 1
	    log_prob = tf.squeeze(dist.log_prob(action))

            score += reward
            agent.remember(observation, action, log_prob, val, reward, mask)
	    observation = observation_

            if update_steps % update_every == 0:
                _, next_value = agent.act(observation_, is_training)
		returns, advantages = agent.generalized_advantage_estimate(next_value)
		agent.update_ppo(returns, advantages, learn_iters, is_training)
		learn_iters +=1

            if update_steps % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i, update_steps, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0
	    
	    if update_steps % save_after == 0:
	        print("Saving parameters")
                agent.save_weights()

	    if done:
	        break

        score_history.append(score)

	print("Logging mean rewards to the file. Episode: {} Reward: {}".format(i, score))
	with test_summary_writer.as_default():
            tf.summary.scalar('reward', score, step=i)

	average_reward.reset_states()

        print_running_reward += score
        print_running_episodes += 1

    env.close()
