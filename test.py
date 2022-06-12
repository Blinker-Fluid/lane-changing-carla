# import sys
# import glob
# import os
import time
import random
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from agents.navigation.controller import VehiclePIDController

from models.DQN import *

if __name__ == '__main__':
    state_height = 45
    state_width = 3
    action_size = 3
    EPISODES = 5
    batch_size = 4
    espisode = 1
    SECONDS_PER_EPISODE = 5 * 60
    count = 0
    
    agent = DQNAgent(state_height, state_width, action_size)
    player = Player()

    ego_speed = player.vehicle.get_velocity()
    ego_lane = player.lane_index

    pos = [ego_speed / 25, 0, 0]
    if ego_lane == LEFT_LANE:
        pos = [ego_speed / 25, 1, 0]
    elif ego_lane == CENTER_LANE:
        pos = [ego_speed / 25, 1, 1]
    elif ego_lane == RIGHT_LANE:
        pos = [ego_speed / 25, 0, 1]
    pos = np.reshape(pos, [1, 3])

    action = 0

    scores = []
    avg_scores = []
    locations = []
    print('Initializing training loop: ')
    for episode in tqdm(range(1, EPISODES + 1), unit='episodes'):
        locations = []
        print('Starting a new episode!')
        player.collision_hist = []

        score = 0
        step = 1

        current_state = player.reset()
        print('Current state: \n', current_state)
        
        done = False
        episode_start = time.time()
        player.do_follow_lane()
        ego_speed = player.vehicle.get_velocity()
        ego_lane = player.lane_index

        pos = [ego_speed / 25, 0, 0]
        if ego_lane == LEFT_LANE:
            pos = [ego_speed / 25, 1, 0]
        elif ego_lane == CENTER_LANE:
            pos = [ego_speed / 25, 1, 1]
        elif ego_lane == RIGHT_LANE:
            pos = [ego_speed / 25, 0, 1]
        pos = np.reshape(pos, [1, 3])
        current_pos = pos
        while True:
            current_state = np.reshape(current_state, [-1, 1, state_height, state_width])
            current_state = np.array(current_state)
            action = agent.act([current_state, current_pos])
            print('action taken: ', action)

            new_state, new_pos, reward, done, _ = player.step(action)

            if action != 0:
                agent.remember1(current_state, action, reward, new_state, done)
            else:
                agent.remember2(current_state, action, reward, new_state, done)

            loc = player.vehicle.get_location()
            locations.append(loc)
            score += reward
            
            current_state = new_state
            print('new state: \n', current_state)
            current_pos = new_pos
            step += 1

            if done:
                agent.save("models/" + str(episode) + ".h5")
                print("weight saved")
                print("episode: {}, epsilon: {}".format(episode, agent.epsilon))
                with open('models/train.txt', 'a') as f:
                    f.write(" episode {} epsilon {}\n".format(episode, agent.epsilon))
                with open('models/trainexp1.pkl', 'wb') as exp1:
                    pickle.dump(agent.memory1, exp1)
                with open('models/exp2.pkl', 'wb') as exp2:
                    pickle.dump(agent.memory2, exp2)

                episode = episode + 1
                if episode == 41:
                    agent.epsilon_min = 0.10
                if episode == 71:
                    agent.epsilon_min = 0.03
                if episode == 6:
                    agent.epsilon_decay = 0.99985  # start epsilon decay
                break

            count += 1
            if count == 10:
                agent.update_target_model()
                print('target model updated')
                count = 0

            if len(agent.memory1) > batch_size and len(agent.memory2) > batch_size:
                agent.replay(batch_size)
            
        gw = player.world.get_map().generate_waypoints(20)
        actor_locations_x = [actor.get_location().x for actor in player.actor_list]
        actor_locations_y = [actor.get_location().y for actor in player.actor_list]

        X_W = [w.transform.location.x for w in gw]
        Y_W = [w.transform.location.y for w in gw]

        plt.figure()
        plt.scatter(X_W, Y_W, c="black", s=0.5)

        X = [loc.x for loc in locations]
        Y = [loc.y for loc in locations]

        plt.plot(X, Y, c="green", linewidth=0.5)
        plt.scatter(X[0], Y[0], c="red", s=1)
        plt.scatter(X[-1], Y[-1], c="blue", s=1)
        plt.scatter(actor_locations_x, actor_locations_y, c="purple", s=1)
        plt.savefig("visualizations/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".png", dpi=300, transparent=True)
        plt.close()

        for actor in player.actor_list:
            actor.destroy()

        scores.append(score)
        avg_scores.append(np.mean(scores[-10:]))

        print('episode: ', episode, 'score %.2f' % score)

    print('Scores List: ', scores)
    print('Avg. Scores List: ', avg_scores)