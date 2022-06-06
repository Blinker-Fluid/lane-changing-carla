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

VEHICLE_VEL = 25
LEFT_LANE = 0
CENTER_LANE = 1
RIGHT_LANE = 2
CELL_LEN = 2.0

class Player():
    def __init__(self, world, bp, vel_ref = VEHICLE_VEL, max_throt = 0.75, max_brake = 0.3, max_steer = 0.8):
        self.world = world
        self.max_throt = max_throt
        self.max_brake = max_brake
        self.max_steer = max_steer
        self.vehicle = None
        self.bp = bp 
        while(self.vehicle is None):
            # spawn_points = world.get_map().get_spawn_points()
            # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            sp = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
                                 carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
            self.vehicle = world.try_spawn_actor(bp, sp)
        
        self.spectator = world.get_spectator()
        self.actor_list = []
        self.lane_index = CENTER_LANE
        
        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 1
        
        self.controller = VehiclePIDController(self.vehicle,
                                        args_lateral=args_lateral_dict,
                                        args_longitudinal=args_longitudinal_dict,
                                        # offset=offset,
                                        max_throttle=max_throt,
                                        max_brake=max_brake,
                                        max_steering=max_steer)
        self.vel_ref = vel_ref
        self.waypointsList = []
        self.current_pos = self.vehicle.get_transform().location
        self.past_pos = self.vehicle.get_transform().location

    def dist_to_waypoint(self, waypoint):
        vehicle_transform = self.vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x)**2 + (vehicle_y - waypoint_y)**2)
    
    def go_to_waypoint(self, waypoint, draw_waypoint = True, threshold = 0.3):
        if draw_waypoint :
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                                       color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                                       persistent_lines=True)
        
        current_pos_np = np.array([self.current_pos.x,self.current_pos.y])
        past_pos_np = np.array([self.past_pos.x,self.past_pos.y])
        waypoint_np = np.array([waypoint.transform.location.x,waypoint.transform.location.y])
        vec2wp = waypoint_np - current_pos_np
        motion_vec = current_pos_np - past_pos_np
        dot = np.dot(vec2wp, motion_vec)
        if (dot >=0):
            while(self.dist_to_waypoint(waypoint) > threshold) :
        
                control_signal = self.controller.run_step(self.vel_ref,waypoint)
                    
                self.vehicle.apply_control(control_signal)
                
                self.update_spectator()

    def get_left_lane_waypoints(self, offset = 2*VEHICLE_VEL):
        # TODO: Check if lane direction is the same as current direction
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        left_lane_target = current_waypoint.get_left_lane().next(offset)[0]
        left_lane_follow = left_lane_target.next(offset)[0]
        self.waypointsList = [left_lane_target, left_lane_follow]

    def get_right_lane_waypoints(self, offset = 2*VEHICLE_VEL):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        right_lane_target = current_waypoint.get_right_lane().next(offset)[0]
        right_lane_follow = right_lane_target.next(offset)[0]
        self.waypointsList = [right_lane_target, right_lane_follow]
    
    def get_current_lane_waypoints(self, offset = 2*VEHICLE_VEL):
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        target_wp = current_waypoint.next(offset)[0]
        follow_wp = target_wp.next(offset)[0]
        self.waypointsList = [target_wp, follow_wp]
    
    def do_left_lane_change(self):
        self.lane_index -= 1
        self.get_left_lane_waypoints()
        for i in range(len(self.waypointsList)-1):
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def do_right_lane_change(self):
        self.lane_index += 1
        self.get_right_lane_waypoints()
        for i in range(len(self.waypointsList)-1):
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()
            
    def do_follow_lane(self):
        self.get_current_lane_waypoints()
        ego_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
        for i in range(len(self.waypointsList)-1):
            # check if we need to slow down
            # set next wp at half the distance to vehicle ahead
            # go there
            # return
            self.current_pos = self.vehicle.get_location()
            self.go_to_waypoint(self.waypointsList[i])
            self.past_pos = self.current_pos
            self.update_spectator()

    def update_spectator(self):
        new_yaw = math.radians(self.vehicle.get_transform().rotation.yaw)
        spectator_transform =  self.vehicle.get_transform()
        spectator_transform.location += carla.Location(x = -10*math.cos(new_yaw), y= -10*math.sin(new_yaw), z = 5.0)
        
        self.spectator.set_transform(spectator_transform)
        self.world.tick()

    def is_waypoint_in_direction_of_motion(self,waypoint):
        current_pos = self.vehicle.get_location()

    def draw_waypoints(self):
        for waypoint in self.waypointsList:
            self.world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                                       color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                                       persistent_lines=True)
    def get_state_representation(self, vehicle_list = []):
        grid = np.ones((51, 3))
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
        for actor in vehicle_list:
            actor_vel = actor.get_velocity()
            actor_speed = (3.6 * math.sqrt(actor_vel.x ** 2 + actor_vel.y ** 2 + actor_vel.z ** 2)) / 100.0
            actor_loc = actor.get_location()
            dist = actor_loc.distance(ego_loc)
            if dist > 60: continue
            
            cell_delta = int(dist // CELL_LEN)

            previous_reference_loc = ego_wp.previous(15)[0].transform.location
            is_behind = actor_loc.distance(previous_reference_loc) <= 15
            if is_behind: cell_delta *= -1

            # find actor's lane
            actor_lane = None
            left_lane_wp = None
            center_lane_wp = None
            right_lane_wp = None

            if self.lane_index == LEFT_LANE:
                left_lane_wp = ego_wp
                center_lane_wp = ego_wp.get_right_lane()
                right_lane_wp = center_lane_wp.get_right_lane()
            elif self.lane_index == CENTER_LANE:
                left_lane_wp = ego_wp.get_left_lane()
                center_lane_wp = ego_wp
                right_lane_wp = ego_wp.get_right_lane()
            elif self.lane_index == RIGHT_LANE:
                left_lane_wp = ego_wp.get_left_lane().get_left_lane()
                center_lane_wp = ego_wp.get_left_lane()
                right_lane_wp = ego_wp

            left_lane_next_wp = left_lane_wp.previous(30)[0]
            center_lane_next_wp = center_lane_wp.previous(30)[0]
            right_lane_next_wp = right_lane_wp.previous(30)[0]

            # TODO: we can speed this up by using "dist" for the range, but we need to handle next/previous search differently
            for i in range(1, 95):
                # print(left_lane_next_wp.transform.location.x, left_lane_next_wp.transform.location.y)
                # print(left_lane_next_wp.transform.location)
                # print(actor_loc.distance(left_lane_next_wp.transform.location))
                if actor_loc.distance(left_lane_next_wp.transform.location) < 1:
                    actor_lane = LEFT_LANE
                    break
                elif actor_loc.distance(center_lane_next_wp.transform.location) < 1:
                    actor_lane = CENTER_LANE
                    break
                elif actor_loc.distance(right_lane_next_wp.transform.location) < 1:
                    actor_lane = RIGHT_LANE
                    break
            
                left_lane_next_wp = left_lane_next_wp.next(1)[0]
                center_lane_next_wp = center_lane_next_wp.next(1)[0]
                right_lane__next_wp = right_lane_next_wp.next(1)[0]
            
            # if we didn't find the actor's lane, it must >30m behind the ego car
            if actor_lane == None:
                continue
            grid[(31 - cell_delta):(35 - cell_delta), actor_lane] = actor_speed
            # TODO: Fill in the grid with actors' velocities
        vel = self.vehicle.get_velocity()
        grid[31:35, self.lane_index] = (3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)) / 100.0

        state = np.zeros((45, 3))
        state[:, :] = grid[3:48, :]
        return state

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