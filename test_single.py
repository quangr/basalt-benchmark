import gym
import minerl
import time
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec
from minerl.herobraine.hero import handlers
# Uncomment to see more logs of the MineRL launch
# import coloredlogs
# coloredlogs.install(logging.DEBUG)
def new_create_observables(self):
    obs_handler_pov = handlers.POVObservation(self.resolution)
    return [
        obs_handler_pov,
        handlers.ObservationFromCurrentLocation(),
        handlers.ObserveFromFullStats("use_item"),
        handlers.ObserveFromFullStats("drop"),
        handlers.ObserveFromFullStats("jump"),
        handlers.ObserveFromFullStats("break_item"),
        handlers.ObserveFromFullStats("craft_item"),
    ]


BasaltBaseEnvSpec.create_observables = new_create_observables


env = gym.make("MineRLBasaltBuildVillageHouse-v0")
env._max_episode_steps = 20
obs = env.reset()
obs = env.reset()

done = False
step = 0
start_time = time.time()

while not done:
    step += 1
    ac = env.action_space.noop()
    # Spin around to see what is around us
    ac["camera"] = [0, 3]
    obs, reward, done, info = env.step(ac)
    env.render()
    
    # Calculate FPS
    if step % 100 == 0:  # Print FPS every 10 steps
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = 100 / elapsed_time
        print(f"FPS: {fps:.2f}")
        start_time = end_time

env.close()