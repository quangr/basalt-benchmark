import gym
import os
import minerl
import time
import videoio

from pyvirtualdisplay import Display
import numpy as np
import concurrent.futures
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec
from minerl.herobraine.hero import handlers
from basalt.adapter import method_dict
import copy
from basalt.config import DefaultDataConfig
from basalt.vpt_lib.tree_util import tree_map
import json
import tyro
from basalt.common import load_model_parameters
from dataclasses import dataclass
from basalt.vpt_lib.agent import MineRLAgent
import torch
from datetime import datetime
import torch_xla.core.xla_model as xm
import torch_xla
from basalt.vpt_lib.agent import resize_image, AGENT_RESOLUTION


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
    # env.seed(39036)
    obs = env.reset()
    obs["pov"] = resize_image(obs["pov"], AGENT_RESOLUTION)
    return obs


# Function to perform a step in the environment
def step_env(env, action):
    obs, reward, done, info = env.step(action)
    if done:
        # env.seed(39036)
        obs = env.reset()
    obs["pov"] = resize_image(obs["pov"], AGENT_RESOLUTION)
    return obs, reward, done, info


# Function to create a new environment
def create_env():
    # env = gym.make("MineRLBasaltFindCave-v0")
    env = gym.make("MineRLBasaltMakeWaterfall-v0")
    env._max_episode_steps = 3000
    return env


# Function to close all environments
def close_envs(envs):
    for env in envs:
        env.close()


def process_observation(results, obs, id, index, format):
    if format == "json":
        obs = copy.copy(obs)  # Create a shallow copy of the observation
        if "pov" in obs:
            del obs["pov"]
            del obs["biome_name"]
            del obs["biome_temperature"]
            del obs["biome_downfall"]
            del obs["sea_level"]
            del obs["life_stats"]["name"]
        if id in results:
            results[id].append(obs)
        else:
            results[id] = [obs]
    else:
        results[index].write(obs["pov"])


NUM_ENVS = 8


@dataclass
class Args:
    data: DefaultDataConfig
    agent_weight: str = "checkpoints/BC/epoch_20.pt"
    w: float = None
    result_format: str = "mp4"


def generate_results_json_path(agent_weight: str, w, suffix="json", i="") -> str:
    # Extract the base directory and filename from the agent_weight path
    base_dir = os.path.dirname(agent_weight)
    weight_filename = os.path.basename(agent_weight)

    # Extract relevant information from the weight filename
    parts = weight_filename.split("_")
    agent_type = parts[0]  # Assuming BC in this case
    epoch = parts[1].split(".")[0]  # Remove file extension

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the results filename
    if w is None:
        results_filename = f"results_{i}_{agent_type}_{epoch}_{timestamp}.{suffix}"
    else:
        results_filename = (
            f"results_{i}_{agent_type}_{epoch}_w={w}_{timestamp}.{suffix}"
        )
    # Combine with the base directory to get the full path
    results_path = os.path.join(base_dir, results_filename)
    if results_path.startswith("/data/"):
        results_path = results_path.replace("/data/", "./")
        directory = os.path.dirname(results_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    return results_path


device = xm.xla_device()


def main(args: Args):
    start_construction_time = time.time()
    ids = list(range(NUM_ENVS))

    # Create environments using multi-threading
    envs = [create_env() for _ in range(NUM_ENVS)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        obs_list = list(executor.map(reset_env, envs))
    end_construction_time = time.time()
    construction_time = end_construction_time - start_construction_time
    print(
        f"Time taken to construct {NUM_ENVS} environments: {construction_time:.2f} seconds"
    )


    done = [False] * NUM_ENVS
    step = 0
    start_time = time.time()

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(
        args.data.model_path
    )

    agent = MineRLAgent(
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(args.data.weights_path)
    agent.policy.eval()

    method_name = os.path.dirname(args.agent_weight).split(os.sep)[-1]
    adapter_dict = method_dict[method_name]
    adapter = adapter_dict["adapter"](agent)
    adapter.load_parameters(args.agent_weight)

    agent_state = agent.policy.initial_state(NUM_ENVS)

    if args.result_format == "json":
        results = {}
    else:
        results = [
            videoio.VideoWriter(
                generate_results_json_path(
                    args.agent_weight, args.w, args.result_format, i
                ),
                resolution=AGENT_RESOLUTION,
                fps=20,
            )
            for i in range(NUM_ENVS)
        ]
    for i in range(NUM_ENVS):
        process_observation(results, obs_list[i], ids[i], i, args.result_format)

    first = torch.ones(NUM_ENVS)
    rollout_num = NUM_ENVS
    while not all(done):
        step += 1
        agent_obs = {
            "img": torch.tensor(
                np.array([[obs_list[i]["pov"]] for i in range(NUM_ENVS)])
            ).to(device)
        }
        with torch_xla.step():
            if args.w is None:
                actions, agent_state = adapter.compute_action(
                    agent_obs, agent_state, first[:, None].bool().to(device)
                )
            else:
                actions, agent_state = adapter.compute_action(
                    agent_obs, agent_state, first[:, None].bool().to(device), args.w
                )
        actions = [
            {key: actions[key][i] for key in actions}
            for i in range(len(actions["attack"]))
        ]
        # Filter environments and actions based on id
        active_envs = [envs[i] for i in range(NUM_ENVS) if ids[i] < rollout_num]
        active_actions = [actions[i] for i in range(NUM_ENVS) if ids[i] < rollout_num]
        active_indices = [i for i in range(NUM_ENVS) if ids[i] < rollout_num]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_list = list(executor.map(step_env, active_envs, active_actions))

        new_obs_list, reward_list, new_done_list, info_list = zip(*results_list)
        for idx, i in enumerate(active_indices):
            assert ids[i] < rollout_num
            if new_done_list[idx]:
                first[i] = True
                new_id = max(ids) + 1
                ids[i] = new_id
                if args.result_format == "mp4":
                    results[i].close()
                    if ids[i] < rollout_num:
                        results[i] = videoio.VideoWriter(
                            generate_results_json_path(
                                args.agent_weight,
                                args.w,
                                args.result_format,
                                ids[i],
                            ),
                            resolution=AGENT_RESOLUTION,
                            fps=20,
                        )
            else:
                first[i] = False
                process_observation(
                    results, obs_list[idx], ids[i], i, args.result_format
                )
                obs_list[i] = new_obs_list[idx]

        if step % 100 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 100 * NUM_ENVS / elapsed_time
            print(f"FPS: {fps:.2f}")
            start_time = end_time

        done = [d or ids[i] >= rollout_num for i, d in enumerate(done)]
    if args.result_format == "json":
        with open(generate_results_json_path(args.agent_weight, args.w), "w") as f:
            json.dump(results, f)
    close_envs(envs)


if __name__ == "__main__":
    disp = Display()
    disp.start()
    args = tyro.cli(Args)
    main(args)
    disp.stop()
