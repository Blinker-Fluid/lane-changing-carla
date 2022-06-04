import sys
import glob
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt 

#The added path depends on where the carla binaries are stored
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import math
import numpy as np
import random
from agents.navigation.controller import VehiclePIDController

VEHICLE_VEL = 15
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
            self.vehicle = world.try_spawn_actor(vehicle_bp, sp)
        
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
            # print(" I draw") 
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
                print(left_lane_next_wp.transform.location.x, left_lane_next_wp.transform.location.y)
                print(left_lane_next_wp.transform.location)
                print(actor_loc.distance(left_lane_next_wp.transform.location))
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

dt = 1.0 / 20.0
args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}

offset = 1

actorList = []
try:
    client = carla.Client("localhost",1555)
    client.set_timeout(10.0)
    world = client.load_world("Town04")

    vehicle_list = []

    spectator = world.get_spectator()
    actorList.append(spectator)
    sp0 = carla.Transform(carla.Location(x=8.526537, y=22.726641, z=0.1),
                        carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
    sp1 = carla.Transform(carla.Location(x=12.526537, y=2.726641, z=0.1),
                        carla.Rotation(pitch=0.000000, yaw=-90.289116, roll=0.000000))
    # spawn_wp = world.get_map().get_waypoint(sp0.location)
    # sp1 = spawn_wp.next(40)[0].transform
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1/20
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("leon")[0]
    

    for i in range(80):
        world.tick()

    player = Player(world, vehicle_bp)
    actorList.append(player.vehicle)
    actorList.append(player.spectator)

    vehicle1 = None
    while vehicle1 is None:
        print("trying to spawn")
        vehicle1 = world.spawn_actor(vehicle_bp, sp0)
    
    vehicle2 = None
    while vehicle2 is None:
        print("trying to spawn")
        vehicle2 = world.spawn_actor(vehicle_bp, sp1)

    vehicle_list.append(vehicle1)
    vehicle_list.append(vehicle2)

    # INIT DQN MODEL DQNAgent = 

    locations = []

    while(True): 
        # CALL MODEL CLASS (DQN)
        player.update_spectator()
        manoeuver = input("Enter manoeuver: ")
        if (manoeuver == "l"): # Perform left lane change
            if (player.lane_index != LEFT_LANE):
                player.do_left_lane_change()
        elif (manoeuver == "r"): # Perform right lane change
            if (player.lane_index != RIGHT_LANE):
                player.do_right_lane_change()
        elif (manoeuver == "f"): # Follow current lane
            player.do_follow_lane()
        elif (manoeuver == "s"): # Skip
            print("skipping")
        elif (manoeuver == "q"):
            break;

        print(player.get_state_representation(vehicle_list = vehicle_list))
        loc = player.vehicle.get_location()
        locations.append(loc)
        print(loc)
        
    gw = world.get_map().generate_waypoints(20)

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

finally:
    print("Destroying actors")
    client.apply_batch([carla.command.DestroyActor(x) for x in actorList])
