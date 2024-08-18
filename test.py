import gym
import minerl
import time
import numpy as np
import concurrent.futures
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS
import copy
import json
# Number of environments
NUM_ENVS = 2

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

# Function to reset environment and return initial observation
def reset_env(env):
    return env.reset()

# Function to perform a step in the environment
def step_env(env, action):
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
    return obs, reward, done, info

# Function to create a new environment
def create_env():
    env = gym.make("MineRLBasaltBuildVillageHouse-v0")
    env._max_episode_steps = 20
    return env

# Function to close all environments
def close_envs(envs):
    for env in envs:
        env.close()

def process_observation(results,obs,id):
    obs = copy.copy(obs)  # Create a shallow copy of the observation
    if 'pov' in obs:
        del obs['pov']
    if id in results:
        results[id].append(obs)
    else:
        results[id]=[obs]

def main():
    results = {}
    start_construction_time = time.time()
    ids = list(range(NUM_ENVS))

    # Create environments using multi-threading
    envs = [create_env() for _ in range(NUM_ENVS)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        obs_list = list(executor.map(reset_env, envs))
    end_construction_time = time.time()
    construction_time = end_construction_time - start_construction_time
    print(f"Time taken to construct {NUM_ENVS} environments: {construction_time:.2f} seconds")

    for i in range(NUM_ENVS):
        process_observation(results, obs_list[i], ids[i])

    done = [False] * NUM_ENVS
    step = 0
    start_time = time.time()

    while not all(done) and max(ids) < 6:
        step += 1
        actions = [env.action_space.noop() for env in envs]
        for action in actions:
            # Spin around to see what is around us
            action["camera"] = [0, 3]

        # Filter environments and actions based on id
        active_envs = [envs[i] for i in range(NUM_ENVS) if ids[i] < 600]
        active_actions = [actions[i] for i in range(NUM_ENVS) if ids[i] < 600]
        active_indices = [i for i in range(NUM_ENVS) if ids[i] < 600]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_list = list(executor.map(step_env, active_envs, active_actions))

        obs_list, reward_list, done_list, info_list = zip(*results_list)

        for idx, i in enumerate(active_indices):
            assert ids[i] < 600
            process_observation(results, obs_list[idx], ids[i])
            if done_list[idx]:
                new_id = max(ids) + 1
                ids[i] = new_id

        if step % 100 == 0:  
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 100 * NUM_ENVS / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = end_time

        done = [d or ids[i] >= 600 for i, d in enumerate(done)]
    with open("test.json", "w") as f:
        # Maximum resident set size (kbytes): 62960
        # f.write(json.dumps(values))
        # Maximum resident set size (kbytes): 46828
        json.dump(results, f)
    close_envs(envs)

if __name__ == "__main__":
    main()
