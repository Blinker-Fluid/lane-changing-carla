import carla
import numpy as np
import random
import math
import cv2

client = carla.Client("localhost", 2223)
world = client.load_world('Town06')
blp_lib = world.get_blueprint_library()
vehicle_bp = blp_lib.filter("cybertruck")[0]
random_spawn_points = random.choice(world.get_map().get_spawn_points()) 
my_vehicle = world.spawn_actor(vehicle_bp, random_spawn_points[0])
vel = vehicle.get_velocity()

# Function to preprocess images from camera sensor
def process_img(self, image):
    i = np.array(image.raw_data)
    #print(i.shape)
    i2 = i.reshape((self.im_height, self.im_width, 4))
    i3 = i2[:, :, :3]
    if self.SHOW_CAM:
        cv2.imshow("", i3)
        cv2.waitKey(1)
    self.front_camera = i3

# Attach a camera sensor
cam_blp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_blp.set_attribute('image_size_x', '1920')
cam_blp.set_attribute('image_size_y', '1080')
cam_blp.set_attribute('fov', '110')
cam_blp.set_attribute('sensor_tick', '1.0')
transform = carla.Transform(carla.Location(x=0.8, z=1.7))
sensor = world.spawn_actor(cam_blp, transform, attach_to=my_vehicle)
sensor.listen(lambda data: process_img(data))
# print(world.get_actors())

# get waypoints 60 meters ahead and 30 meters behind for three lanes
waypoints_left_front = waypoint.get_left_lane(60)
waypoints_center_front = waypoint.next(60)
waypoints_right_front = waypoint.get_right_lane(60)
waypoints_left_behind = waypoint.get_left_lane(30)
waypoints_center_behind = waypoint.previous(30)
waypoints_right_behind = waypoint.get_right_lane(30)

# get ego car's speed
car_speed = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) # in Km/h

# get ego car's location and divide by 4 to be filled in grid
ego_car_lane = my_vehicle.get_location() // 4 # IN REFERENCE THEY USE ('d') FRENET COORDINATE INSTEAD 

# Get locations of waypoints
left_waypoints_locations = []
center_waypoints_locations = []
right_waypoints_locations = []

# Appending waypoints locations to the above lists 
for point in waypoints_left_behind:
    left_waypoints_locations.append(point.transform.location)
for point in waypoints_center_behind:
    center_waypoints_locations.append(point.transform.location)
for point in waypoints_right_behind:
    right_waypoints_locations.append(point.transform.location)
for point in waypoints_left_front:
    left_waypoints_locations.append(point.transform.location)
for point in waypoints_center_front:
    center_waypoints_locations.append(point.transform.location)
for point in waypoints_right_front:
    right_waypoints_locations.append(point.transform.location)

# Getting actors locations
actor_list = world.get_actors()
actors_locations = []
for actor in actor_list:
    actor_locations.append(actor.get_location())

### Check if locations of waypoints and actors overlap? (TBD)

# State representation matrix size
state_height = 45
state_width = 3
action_size = 3

# create empty grid
grid = np.ones((51, 3))

# ego_car_lane = my_vehicle.get_location().y // 4 # ???
grid[31:35, ego_car_lane] = car_speed / 100.0 # fill and normalize

### IGNORED LINES 219 -> 230 IN REFERENCE'S TRAIN.PY FOR NOW!

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