import carla
import numpy as np
import random
import math
import cv2
import matplotlib.pyplot as plt

client = carla.Client("localhost", 1506)
world = client.load_world('Town04')
world = client.get_world()
print(world.get_map())
gw = world.get_map().generate_waypoints(20)

X_W = [w.transform.location.x for w in gw]
Y_W = [w.transform.location.y for w in gw]

plt.figure()
plt.scatter(X_W, Y_W, c="black", s=1)

blp_lib = world.get_blueprint_library()
vehicle_bp = blp_lib.filter("cybertruck")[0]
sp = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
                     carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
my_vehicle = world.spawn_actor(vehicle_bp, sp)
vel = my_vehicle.get_velocity()

# Function to preprocess images from camera sensor
# im_width = 640
# im_height = 480
# SHOW_CAM = False

# def process_img(image):
#     i = np.array(image.raw_data)
#     #print(i.shape)
#     i2 = i.reshape((im_height, im_width, 4))
#     i3 = i2[:, :, :3]
#     if SHOW_CAM:
#         cv2.imshow("", i3)
#         cv2.waitKey(1)
#     front_camera = i3

# Attach a camera sensor
# cam_blp = world.get_blueprint_library().find('sensor.camera.rgb')
# cam_blp.set_attribute('image_size_x', f'{im_width}')
# cam_blp.set_attribute('image_size_y', f'{im_height}')
# cam_blp.set_attribute('fov', '110')
# cam_blp.set_attribute('sensor_tick', '1.0')
# transform = carla.Transform(carla.Location(x=0.8, z=1.7))
# sensor = world.spawn_actor(cam_blp, transform, attach_to=my_vehicle)
# sensor.listen(lambda data: process_img(data))
# print(world.get_actors())

# get waypoints 60 meters ahead and 30 meters behind for three lanes
current_lane_waypoint = world.get_map().get_waypoint(my_vehicle.get_location())
left_lane_waypoint = current_lane_waypoint.get_left_lane()
right_lane_waypoint = current_lane_waypoint.get_right_lane()

current_lane_waypoints = [current_lane_waypoint]
left_lane_waypoints = [left_lane_waypoint]
right_lane_waypoints = [right_lane_waypoint]

for i in range(500):
    if i < 0:
        current_lane_waypoints.append(
            current_lane_waypoint.previous(-i * 2)[0])
        left_lane_waypoints.append(left_lane_waypoint.previous(-i * 2)[0])
        right_lane_waypoints.append(right_lane_waypoint.previous(-i * 2)[0])

    elif i >= 0:
        current_lane_waypoints.append(
            current_lane_waypoint.next((i + 1) * 2)[0])
        # left_lane_waypoints.append(left_lane_waypoint.next((i + 1) * 2)[0])
        # right_lane_waypoints.append(right_lane_waypoint.next((i + 1) * 2)[0])

X_C = [w.transform.location.x for w in current_lane_waypoints]
Y_C = [w.transform.location.y for w in current_lane_waypoints]

X_L = [w.transform.location.x for w in left_lane_waypoints]
Y_L = [w.transform.location.y for w in left_lane_waypoints]

X_R = [w.transform.location.x for w in right_lane_waypoints]
Y_R = [w.transform.location.y for w in right_lane_waypoints]

# plt.figure()
plt.scatter(X_C, Y_C, c="yellow", s=0.2)
plt.scatter(X_L, Y_L, c="blue", s=0.2)
plt.scatter(X_R, Y_R, c="red", s=0.2)
plt.scatter(current_lane_waypoint.transform.location.x,
            current_lane_waypoint.transform.location.x, c="green", s=20)
plt.savefig("test.png")
plt.close()

# get ego car's speed
car_speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)  # in Km/h

# get ego car's location and divide by 4 to be filled in grid
# ego_car_lane = my_vehicle.get_location() // 4 # IN REFERENCE THEY USE ('d') FRENET COORDINATE INSTEAD
# NOTE: We don't need the frenet coordinates, we have the waypoints already. We just iterate over them.

# Getting actors locations
actor_list = world.get_actors()
# actors_locations = []
# for actor in actor_list:
#     actor_locations.append(actor.get_location())

# Check if locations of waypoints and actors overlap? (TBD)
# get all actors
# filter for vehicles
# 

# State representation matrix size
state_height = 45
state_width = 3
action_size = 3

# create empty grid
grid = np.ones((51, 3))

# ego_car_lane = my_vehicle.get_location().y // 4 # ???
# grid[31:35, ego_car_lane] = car_speed / 100.0 # fill and normalize

# IGNORED LINES 219 -> 230 IN REFERENCE'S TRAIN.PY FOR NOW!

# Creating and filling out state representation grid
state = np.zeros((state_height, state_width))
state[:, :] = grid[3:48, :]
state = np.reshape(state, [-1, 1, state_height, state_width])
pos = [car_speed / 50, 0, 0]
if ego_car_lane == 0:
    pos = [car_speed / 50, 0, 1]
elif ego_car_lane == 1:
    pos = [car_speed / 50, 1, 1]
elif ego_car_lane == 2:
    pos = [car_speed / 50, 1, 0]
pos = np.reshape(pos, [1, 3])
