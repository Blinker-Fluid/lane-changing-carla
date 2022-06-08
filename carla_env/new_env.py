import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math

from agents.navigation.controller import VehiclePIDController

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

VEHICLE_VEL = 5
LEFT_LANE = 0
CENTER_LANE = 1
RIGHT_LANE = 2
CELL_LEN = 2.0
STEER_AMT = 1.0
offset = 1

class Env():
    # state = None
    def __init__(self):
        self.client = carla.Client("localhost", 1555)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town04")
        self.blp_lib = self.world.get_blueprint_library()
        self.model_3 = self.blp_lib.filter("model3")[0]
        self.spectator = self.world.get_spectator()

    def reset(self):
        self.actor_list = []
        self.collision_hist = []
        self.vehicle_list = []

        self.transform = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
                                         carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
        self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
        
        self.world.tick()

        self.sp0 = random.choice(self.world.get_map().get_spawn_points())
        self.sp1 = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle1 = None
        while self.vehicle1 is None:
            print("Spawning vehicle 1")
            self.vehicle1 = self.world.spawn_actor(self.model_3, self.sp0)
        self.vehicle2 = None
        while self.vehicle2 is None:
            print("Spawning vehicle 2")
            self.vehicle2 = self.world.spawn_actor(self.model_3, self.sp1)
        
        self.vehicle_list.append(self.vehicle)
        self.vehicle_list.append(self.vehicle1)
        self.vehicle_list.append(self.vehicle2)
        
        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.spectator)

        # spawning at a specific location for debugging
        # while(self.vehicle is None):
        #     # spawn_points = world.get_map().get_spawn_points()
        #     # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        #     sp = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
        #                          carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
        #     self.vehicle = world.try_spawn_actor(vehicle_bp, sp)
        
        sp = carla.Transform(carla.Location(x=12.258620, y=68.035088, z=0.281942),
                             carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
        

        self.lane_index = CENTER_LANE
        self.vel_ref = VEHICLE_VEL
        self.waypointsList = []
        self.current_pos = self.vehicle.get_transform().location
        self.past_pos = self.vehicle.get_transform().location
        
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 1/20
        self.world.apply_settings(self.settings)

        # collision sensor
        col_sensor = self.blp_lib.find('sensor.other.collision')
        col_transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        self.ego_col = self.world.spawn_actor(col_sensor, col_transform, attach_to=self.vehicle)
        self.ego_col.listen(lambda event: self.collision_data(event))

        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        # self.state_rep = get_state_representation(vehicle_list = self.vehicle_list)

        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 1
        max_throt = 0.75
        max_brake = 0.3
        max_steer = 0.8

        self.controller = VehiclePIDController(self.vehicle,
                                        args_lateral=args_lateral_dict,
                                        args_longitudinal=args_longitudinal_dict,
                                        # offset=offset,
                                        max_throttle=max_throt,
                                        max_brake=max_brake,
                                        max_steering=max_steer)
        
        grid = np.ones((51, 3))
        # ego_loc = self.vehicle.get_location()
        ego_loc = self.vehicle.get_location()
        ego_wp = self.world.get_map().get_waypoint(ego_loc)
        
        for actor in self.vehicle_list:
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

        # while self.state is None:
        #     time.sleep(0.01)

        self.state = np.zeros((45, 3))
        self.state[:, :] = grid[3:48, :]
        
        return self.state

    def collision_data(event):
            self.collision_hist.append(event)
            print("Collision detected:\n" + str(self.collision_data) + '\n')

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
    # def get_state_representation(self, vehicle_list = []):
        

    def step(self, action):
        # if action == 0:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        # elif action == 1:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer= 0))
        # elif action == 2:
        #     self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        SECONDS_PER_EPISODE = 10
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

        return self.state, reward, done, None