import carla

client = carla.Client("localhost", 2223)
world = client.load_world('Town01')
blp_lib = world.get_blueprint_library()
vehicle_bp = blp_lib.find("vehicle.lincoln.mkz2017")
spawn_points = world.get_map().get_spawn_points() 
my_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])

print(world.get_actors())