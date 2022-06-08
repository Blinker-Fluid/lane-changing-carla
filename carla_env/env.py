import sys
import glob
import os
import time
import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from agents.navigation.controller import VehiclePIDController

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

# from func_timeout import FunctionTimedOut, func_timeout
# from utils.carla_server import kill_carla_server, start_carla_server

# from carla_env.managers.actor_manager import ActorManager
# from carla_env.managers.observation_manager import ObservationManager
# from carla_env.managers.plan_manager import PlanManager
# from carla_env.managers.reward_manager import RewardManager
# from carla_env.managers.sensor_manager import SensorManager
# from carla_env.managers.weather_manager import WeatherManager

# TODO: prune, or delete if not necessary!
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 1555)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #print(i.shape)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

class Env():
    def __init__(self, config=None):
        # Variables that do not change between episodes
        self.fps = config["fps"]
        self.random_maps = config["random_maps"]
        self.route_dist_limit = config["route_dist_limit"]

        # Start server
        start_carla_server(config["server_port"])
        # Create client and traffic manager instances
        self.client = None
        self.traffic_manager = None
        self.make_client(config["server_port"])
        self.make_tm()

        blp_lib = self._world.get_blueprint_library()
        # Create managers for various aspects of the environment
        self._sensor_manager = SensorManager(blp_lib, config["sensors"])
        self._reward_manager = RewardManager()
        self._actor_manager = ActorManager(
            self.client, blp_lib, config["num_walkers"], config["num_vehicles"],
            config["ego_spawn_point_idx"])
        self._weather_manager = WeatherManager(config["dynamic_weather"], 1 / self.fps)
        self._plan_manager = PlanManager(
            config["dense_wp_interval"], config["sparse_wp_interval"], config["debug"])
        self._obs_manager = ObservationManager(config["features"], config["speed_limit"])

        # Counter for episodes
        self.episode_counter = 0

        # Variables that change for every episode
        self._world = self.client.load_world(config["map"])
        self._opendrive_map = self._world.get_map()
        self._ego_vehicle = None

        # Set synchronous mode for client and tm
        self._set_synchronous_mode(True)

        # Variables that change during episodes
        self.current_step = -1

    def make_client(self, server_port):
        """Create client and world for the environment. Called in __init__"""
        client_is_initialized = False
        # This sleep is to wait until the carla server is up and running.
        # Otherwise we print an error
        sleep(4)
        print("Creating client")
        counter = 0
        while not client_is_initialized:
            try:
                counter += 1
                self.client = carla.Client("localhost", server_port)
                self.client.set_timeout(20.0)
                self._world = self.client.get_world()
                client_is_initialized = True
            except RuntimeError as err:
                if counter > 3:
                    print(err)
                    print("Trying again...")

    def make_tm(self):
        print("Creating tm")
        tm_port = 9500
        tm_is_initialized = False
        while not tm_is_initialized:
            try:
                self.traffic_manager = self.client.get_trafficmanager(tm_port)
                tm_is_initialized = True
            except Exception as err:
                print("Caught exception during traffic manager creation: ")
                print(err)
                tm_port += 1
                print("Trying with port {}...".format(tm_port))

    def reset(self):
        """Resets the environment."""
        self.episode_counter += 1

        self._cleanup()
        if self.random_maps:
            new_map = random.choice(self.client.get_available_maps())
            self._world = self.client.load_world(new_map)
            self._opendrive_map = self._world.get_map()
            self._set_synchronous_mode(True)

        self._actor_manager.reset(self._world, self._opendrive_map)
        self._ego_vehicle = self._actor_manager.spawn_ego_vehicle()
        self._sensor_manager.reset(self._world, self._ego_vehicle)
        self._weather_manager.reset(self._world)
        self._plan_manager.reset(self._world, self._ego_vehicle, self._opendrive_map)
        self._obs_manager.reset(self._world, self._ego_vehicle, self._opendrive_map)

        self.current_step = -1

        self._actor_manager.spawn_vehicles()
        self._actor_manager.spawn_walkers()
        self._sensor_manager.spawn_sensors(self._world)

        self._move_spectator()

        # Commands may not register in the first few seconds, so we skip them
        for i in range(self.fps * 2):
            current_frame = self._world.tick()

        sensor_data = self._sensor_manager.tick(current_frame)
        dense_target, sparse_target = self._plan_manager.step(sensor_data["gps"])
        prev_dense_target = self._plan_manager.prev_dense_target()
        state = self._obs_manager.get_state(dense_target, sparse_target, prev_dense_target)
        state.update(sensor_data)

        fake_action = {
            "throttle": 0,
            "brake": 0,
            "steer": 0
        }
        reward_dict = self._reward_manager.get_reward(state, fake_action)
        is_terminal = self._get_terminal(state)
        return state, reward_dict, is_terminal

    def step(self, action):
        self.current_step += 1
        control = carla.VehicleControl(
            throttle=action["throttle"],
            brake=action["brake"],
            steer=action["steer"]
        )
        self._ego_vehicle.apply_control(control)
        self._move_spectator()

        current_frame = self._world.tick()
        self._weather_manager.tick()
        self._actor_manager.update_lights(self._weather_manager.weather)

        sensor_data = self._sensor_manager.tick(current_frame)
        dense_target, sparse_target = self._plan_manager.step(sensor_data["gps"])
        prev_dense_target = self._plan_manager.prev_dense_target()
        state = self._obs_manager.get_state(dense_target, sparse_target, prev_dense_target)
        state.update(sensor_data)

        reward_dict = self._reward_manager.get_reward(state, action)
        is_terminal = self._get_terminal(state)
        return state, reward_dict, is_terminal

    def _get_terminal(self, state):
        # is_terminal is either empty, or contains our cause for termination as a str
        is_terminal = []
        if state["collision"]:
            is_terminal.append("collision")
            print("Collision occured.", " " * 40)
        if self._plan_manager.is_route_completed():
            is_terminal.append("finished")
            print("Reached last waypoint.", " " * 40)
        if abs(state["route_dist"]) > self.route_dist_limit:
            is_terminal.append("route_dist")
            print("Got too far from lane center.", " " * 40)

        return is_terminal

    def _set_synchronous_mode(self, sync):
        """Set or unset synchronous mode for the server and the traffic manager."""
        settings = self._world.get_settings()
        settings.synchronous_mode = sync
        if sync:
            settings.fixed_delta_seconds = 1 / self.fps
        else:
            settings.fixed_delta_seconds = None
        self._world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(sync)

    def _move_spectator(self):
        """Move simulator camera to vehicle for viewing."""
        spectator = self._world.get_spectator()
        transform = self._ego_vehicle.get_transform()
        transform.location.z += 20
        transform.rotation.pitch = -90
        transform.rotation.roll = 0
        transform.rotation.yaw = 0
        spectator.set_transform(transform)

    def _cleanup(self):
        """Destroy leftover actors."""
        self._sensor_manager.cleanup()
        self._actor_manager.cleanup()
        self._obs_manager.cleanup()
        self._world.tick()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            print("Exiting...")
            func_timeout(10, self._cleanup)
            func_timeout(10, self._set_synchronous_mode, (False,))
            kill_carla_server()
        except FunctionTimedOut:
            print("Timeout while attempting to set CARLA to async mode.")
        except Exception as err:
            print(err)