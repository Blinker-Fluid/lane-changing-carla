import sys
import glob
import os
import time
import random
# import math
# import pickle
# import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# from agents.navigation.controller import VehiclePIDController

# from models.DQN import *

#The added path depends on where the carla binaries are stored
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

client = carla.Client("localhost", 1555)
client.set_timeout(10.0)
world = client.load_world("Town05")
# # Spawning ego car

spawn_wp = world.get_map().get_waypoint(carla.Location(x=0.3564453125, y=-204.24703979492188))
print(spawn_wp.transform.location.x)
print(spawn_wp.transform.location.y)
print(spawn_wp.transform.location.z)
blp_lib = world.get_blueprint_library()
model_3 = blp_lib.filter("model3")[0]
# # Spawning ego car
spectator = world.get_spectator()
initial_transform = carla.Transform(carla.Location(x=0.3564453125, y=-204.24703979492188, z=0.5),
                                  carla.Rotation(pitch=0.000000, yaw=179.757080078125, roll=0.000000))

vehicle = world.spawn_actor(model_3, initial_transform)

gw = world.get_map().generate_waypoints(10)
X_W = [w.transform.location.x for w in gw]
Y_W = [w.transform.location.y for w in gw]

plt.figure()
plt.scatter(X_W, Y_W, c="black", s=0.5)
plt.scatter(vehicle.get_transform().location.x, vehicle.get_transform().location.y, s=2, c="red")
plt.savefig(datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".png", dpi=300, transparent=True)
plt.close()
