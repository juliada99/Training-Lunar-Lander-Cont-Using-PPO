import os
import glob
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import gym

from ppo_agent import Agent



def test():

    print("============================================================================================")

    ################## hyperparameters ##################

    total_test_episodes = 20
    render = True
    frame_delay = 0.01

    env_name = "LunarLanderContinuous-v2"

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]
    print("state dim: ", state_dim)

    # action space dimension
    action_dim = env.action_space.shape[0]
    print("action dim: ", action_dim)

    update_every = 20

    minibatch_size = 5
    n_epochs = 4
    alpha = 0.0003  # learning_rate
    gamma = 0.99
    lam = 0.95
    save_path = '/home/rwl/Desktop/Lunar_Lander_PPO'

    test_log_dir = save_path + '/logs/test' 
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    # initialize a PPO agent
    agent = Agent(state_dim, action_dim, 0.2, 0.5, minibatch_size, update_every, n_epochs, gamma, lam, alpha, save_path, test_summary_writer)


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num


    agent.load_weights()

    print("--------------------------------------------------------------------------------------------")
    
    max_ep_len = 1000

    is_training = False
    test_running_reward = 0


    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = agent.act(state, is_training)
            state, reward, done, _ = env.step(action.numpy())
            ep_reward += reward

            if render:
		# display to the screen
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
	if agent.batch_data.is_full():
            agent.batch_data.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
	with test_summary_writer.as_default():
            tf.summary.scalar('reward', ep_reward, step=ep)



    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")



if __name__ == '__main__':

    test()


