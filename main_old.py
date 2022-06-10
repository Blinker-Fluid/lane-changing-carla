import sys
import glob
import os
import time
import random
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
from keras.callbacks import TensorBoard

from models.DQN import *
from carla_env.new_env import *

if __name__ == '__main__':

    state_height = 45
    state_width = 3
    action_size = 3
    EPISODES = 100
    epsilon = 1
    batch_size = 16
    espisode = 1
    # MIN_REWARD = -200
    # EPSILON_DECAY = 0.95 ## 0.9975 99975
    # MIN_EPSILON = 0.001
    # AGGREGATE_STATS_EVERY = 10

    agent = DQNAgent(state_height, state_width, action_size)
    env = Env()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=60)

    # Loop over episodes
    while True:
        for i in range(80):
            env.world.tick()

        print('Restarting episode')

        # Reset environment and get initial state
        state = env.reset()
        env.collision_hist = []
        locations = []

        done = False

        # Loop over steps
        while True:
            # For FPS counter
            step_start = time.time()

            # Show current state
            print(state)
            state = np.reshape(state, [-1, 1, state_height, state_width])

            env.update_spectator()
            action = agent.act(state)
            if (action == 0): # Follow current lane
                env.do_follow_lane()
            elif (action == 1): # Perform left lane change
                if (env.lane_index != LEFT_LANE):
                    env.do_left_lane_change()
            elif (action == 2): # Perform right lane change
                if (env.lane_index != RIGHT_LANE):
                    env.do_right_lane_change()
            
            loc = env.vehicle.get_location()
            locations.append(loc)

            # Step environment (additional flag informs environment to not break an episode by time limit)
            new_state, reward, done, _ = env.step(action)

            # Set current step for next loop iteration
            state = new_state

            # If done - agent crashed, break an episode
            if done:
                break

            # Measure step time, append to a deque, then print mean FPS for last 60 frames, q values and taken action
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)
           
        print('finished an episode!')

        gw = env.world.get_map().generate_waypoints(20)

        X_W = [w.transform.location.x for w in gw]
        Y_W = [w.transform.location.y for w in gw]

        plt.figure()
        plt.scatter(X_W, Y_W, c="black", s=1)
        
        X = [loc.x for loc in locations]
        Y = [loc.y for loc in locations]

        plt.scatter(X, Y, c="green", s=1)
        plt.scatter(X[0], Y[0], c="red", s=2)
        plt.scatter(X[-1], Y[-1], c="blue", s=2)
        plt.savefig("visualizations/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".png", dpi=300)
        plt.close()

        # Destroy an actor at end of episode
        print("Destroying actors")
        env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])

    # FPS = 15
    # ep_rewards = [-200]

    # # For more repetitive results
    # random.seed(1)
    # np.random.seed(1)
    # tf.compat.v1.set_random_seed(1)
    
    # # Create models folder
    # if not os.path.isdir('models_trained'):
    #     os.makedirs('models_trained')

    # state_height = 45
    # state_width = 3
    # action_size = 3
    # EPISODES = 100
    # epsilon = 1
    # batch_size = 16
    # espisode = 1
    # # MIN_REWARD = -200
    # # EPSILON_DECAY = 0.95 ## 0.9975 99975
    # # MIN_EPSILON = 0.001
    # # AGGREGATE_STATS_EVERY = 10

    # agent = DQNAgent(state_height, state_width, action_size)
    # env = Env()

    # while episode <= EPISODES:
    #     env.update_spectator()
    #     state = env.reset()
    #     action = 0
    #     count = 0
        
    #     last_act = action
    #     if env.collision_hist[-1] == True:
    #         last_reward = -30
        
    #     if 

    #     model_output = agent.act(state)
    #     if (model_output == 0): # Follow current lane
    #         env.do_follow_lane()
    #     elif (model_output == 1): # Perform left lane change
    #         if (env.lane_index != env.LEFT_LANE):
    #             env.do_left_lane_change()
    #     elif (model_output == 2): # Perform right lane change
    #         if (env.lane_index != env.RIGHT_LANE):
    #             env.do_right_lane_change()

    #     print(env.state)
    #     loc = env.vehicle.get_location()
    #     locations.append(loc)
    #     print(loc)
        
    #     done = False
    #     if done == True:
    #         agent.save("./models_trained/episode" + str(episode) + ".h5")
    #         print("weights saved")
    #         print("episode: {}, epsilon: {}".format(episode, agent.epsilon))
    #         with open("./models_trained/train.txt", "a") as f:
    #             f.write(" episode {} epsilon {}\n".format(episode, agent.epsilon))
    #         with open('./models_trained/trainexp1.pkl', 'wb') as exp1:
    #             pickle.dump(agent.memory1, exp1)
    #         with open('./models_trained/exp2.pkl', 'wb') as exp2:
    #             pickle.dump(agent.memory2, exp2)
    #         episode = episode + 1
    #         if episode == 41:
    #             agent.epsilon_min = 0.10
    #         if episode == 71:
    #             agent.epsilon_min = 0.03
    #         if episode == 6:
    #             agent.epsilon_decay = 0.99985  # start epsilon decay
    #         break

        

        



#NOTE: SENTDEX'S IMPLEMENTATION:
    # # Iterate over episodes
    # scores = []
    # avg_scores = []
    # for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    #     env.collision_hist = []

    #     # Restarting episode - reset episode reward and step number
    #     score = 0
    #     step = 1

    #     # Reset environment and get initial state
    #     current_state = env.reset()

    #     # Reset flag and start iterating until episode ends
    #     done = False
    #     episode_start = time.time()

    #     # Play for given number of seconds only
    #     while True:

    #         # This part stays mostly the same, the change is to query a model for Q values
    #         if np.random.random() > epsilon:
    #             # Get action from Q table
    #             action = np.argmax(agent.act(current_state))
    #         else:
    #             # Get random action
    #             action = np.random.randint(0, 3)
    #             # This takes no time, so we add a delay matching 60 FPS (prediction above takes longer)
    #             time.sleep(1/FPS)

    #         new_state, reward, done, _ = env.step(action)

    #         # Transform new continous state to new discrete state and count reward
    #         score += reward

    #         # Every step we update replay memory
    #         agent.remember1((current_state, action, reward, new_state, done))

    #         current_state = new_state
    #         step += 1

    #         model_output = agent.act(current_state)
    #         if (model_output == 0): # Follow current lane
    #             env.do_follow_lane()
    #         elif (model_output == 1): # Perform left lane change
    #             if (env.lane_index != LEFT_LANE):
    #                 env.do_left_lane_change()
    #         elif (model_output == 2): # Perform right lane change
    #             if (env.lane_index != RIGHT_LANE):
    #                 env.do_right_lane_change()

    #         print(current_state)
    #         loc = env.vehicle.get_location()
    #         locations.append(loc)
    #         print(loc)

    #         if done:
    #             break
                
    #     gw = world.get_map().generate_waypoints(20)

    #     X_W = [w.transform.location.x for w in gw]
    #     Y_W = [w.transform.location.y for w in gw]

    #     plt.figure()
    #     plt.scatter(X_W, Y_W, c="black", s=1)

    #     X = [loc.x for loc in locations]
    #     Y = [loc.y for loc in locations]

    #     plt.scatter(X, Y, c="green", s=1)
    #     plt.scatter(X[0], Y[0], c="red", s=2)
    #     plt.scatter(X[-1], Y[-1], c="blue", s=2)
    #     plt.savefig("visualizations/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".png", dpi=300)
    #     plt.close()

    #     # End of episode - destroy agents
    #     for actor in env.actor_list:
    #         actor.destroy()

    #     scores.append(score)
    #     avg_scores.append(np.mean(scores[-10:]))

    #     if not episode % AGGREGATE_STATS_EVERY or episode == 1:
    #         avg_scores.append(np.mean(scores[-AGGREGATE_STATS_EVERY:]))
    #         min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #         max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #         agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)


    #             #Save model, but only when min reward is greater or equal a set value
    #         if min_reward >= MIN_REWARD:
    #             agent.model.save(f'models_trained/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_score:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    #         print('episode: ', episode, 'score %.2f' % score)
    #         # Decay epsilon
    #         if epsilon > MIN_EPSILON:
    #             epsilon *= EPSILON_DECAY
    #             epsilon = max(MIN_EPSILON, epsilon)

    # # Set termination flag for training thread and wait for it to finish
    # agent.terminate = True
    # # trainer_thread.join()
    # agent.model.save(f'models_trained/{MODEL_NAME}__{max_reward:_>7.2f}max_{avg_score:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(scores)
    # plt.plot(avg_scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()