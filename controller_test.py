import carla
import numpy as np
import random
import math
import cv2
import matplotlib.pyplot as plt
from carla_env.controller import VehiclePIDController

FPS = 12
dt = 1.0 / 20.0
target_speed = 20.0  # Km/h
sampling_radius = 2.0
args_lateral = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
args_longitudinal = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
max_throt = 0.75
max_brake = 0.3
max_steer = 0.8
offset = 0
base_min_distance = 3.0
follow_speed_limits = False


client = carla.Client("localhost", 1506)
world = client.load_world('Town04')
world = client.get_world()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1 / FPS
world.apply_settings(settings)


blp_lib = world.get_blueprint_library()
vehicle_bp = blp_lib.filter("cybertruck")[0]
sp = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
                     carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
my_vehicle = world.spawn_actor(vehicle_bp, sp)
controller = VehiclePIDController(my_vehicle, args_lateral, args_longitudinal)


# # get waypoints ahead
# current_waypoint = world.get_map().get_waypoint(my_vehicle.get_location())
# next_waypoint = current_waypoint.next(5)[0]

for i in range(FPS * 4):
    world.tick()

location_samples = []

current_waypoint = world.get_map().get_waypoint(my_vehicle.get_location())
next_waypoint = current_waypoint.next(15)[0]
action = 0


for i in range(FPS * 20):
    location = my_vehicle.get_location()
    print(location)
    location_samples.append(location)

    if i == FPS * 5:
        action = 1

    current_waypoint = world.get_map().get_waypoint(location)

    if action == 0:
        if current_waypoint == next_waypoint:
            next_waypoint = current_waypoint.next(15)[0]
    if action == 1:
        if current_waypoint != next_waypoint:
            next_waypoint = current_waypoint.get_right_lane().next(15)[0]
        else:
            next_waypoint = current_waypoint.next(15)[0]
            action = 0

    control = controller.run_step(target_speed, next_waypoint)
    my_vehicle.apply_control(control)
    world.tick()

gw = world.get_map().generate_waypoints(20)

X_W = [w.transform.location.x for w in gw]
Y_W = [w.transform.location.y for w in gw]

plt.figure()
plt.scatter(X_W, Y_W, c="black", s=1)

X = [loc.x for loc in location_samples]
Y = [loc.y for loc in location_samples]


plt.scatter(X, Y, c="green", s=1)
plt.scatter(X[0], Y[0], c="red", s=2)
plt.scatter(X[-1], Y[-1], c="blue", s=2)
plt.savefig("test.png", dpi=300)
plt.close()
