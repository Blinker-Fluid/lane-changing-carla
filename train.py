from models.DQN import *
from carla_env.env import *

# Actors
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

    # Training Loop
    EPISODES = 100
    # state_height = 45
    # state_width = 3
    state = player.get_state_representation(vehicle_list = vehicle_list)
    action_size = 3
    batch_size = 16
    episode = 1
    action = 0
    count = 0
    start = time.time()
    locations = []

    # NOTE: not required, but kept for reference if needed.
    # state = np.reshape(state, [-1, 1, state.shape[0], state.shape[1]])
    # print(state.shape)
    # pos = [player.vehicle.get_velocity() / 50, 0, 0]
    # print(pos)

    while(True): 
        state = player.get_state_representation(vehicle_list = vehicle_list)
        print(state.shape)
        agent = DQNAgent(state.shape[0], state.shape[1], action_size)
        player.update_spectator()

        # TODO: implement this at the end.
        # if done:
            # destroy actors
            # compute reward: 
            # save model: (but only when min reward is greater or equal a set value) agent.save("./train/episode" + str(episode) + ".h5")
            # decay epsilon:
                # episode = episode + 1
                # if episode == 41:
                #     agent.epsilon_min = 0.10
                # if episode == 71:
                #     agent.epsilon_min = 0.03
                # if episode == 6:
                #     agent.epsilon_decay = 0.99985  # start epsilon decay
                # break
            # save model after training
            # plot scores

        # TODO: Reward Calculation
        # last_action = action

        # if last_action == 0:
        #     last_reward = (2 * (()))

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
            break

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