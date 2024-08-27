import torch_xla.runtime as xr

xr.use_spmd()

import os
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from basalt.utils.chunk_loader import (
    BasaltMinecraftDataset,
    data_generator,
    get_mp4_tuple,
)
from basalt.vpt_lib.tree_util import tree_map
from basalt.adapter import method_dict
import glob
from torch.utils.data import DataLoader
import time
import webdataset as wds
import torch_xla.core.xla_model as xm

# torch_xla.experimental.eager_mode(True)
import torch_xla

# device = torch.device("cuda")
device = xm.xla_device()
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh


@dataclass
class Args:
    in_model: str = "/data/foundation-model-3x.model"
    in_weights: str = "/data/foundation-model-3x.weights"


def train(args: Args):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(args.in_model)

    agent = MineRLAgent(
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(args.in_weights)
    policy = agent.policy
    policy.eval()
    # policy = torch.compile(policy, backend="openxla")

    dataset_dict = {
        "/data/demonstrations/MineRLBasaltFindCave-v0": 0,
        # "/data/demonstrations/MineRLBasaltMakeWaterfall-v0": 1,
    }

    need_embed = {}
    for dataset_dir, label in dataset_dict.items():
        task_name = (
            os.path.basename(dataset_dir).split("-")[0].replace("MineRLBasalt", "")
        )

        file_list_path = f"scripts/filelists/{task_name}_urls.txt"
        with open(file_list_path, "r") as f:
            file_list = f.read().splitlines()
        end = False
        zip_size = ((len(file_list) // 75) // 2) * 2
        for i in range(0, len(file_list) // (zip_size)):
            tar_file = f"{i}.tar"
            tar_path = os.path.join(dataset_dir, tar_file)

            if not os.path.exists(tar_path):
                need_embed[tar_path] = []
                group = file_list[i * zip_size : i * zip_size + zip_size]
                for file in group:
                    file_path = os.path.join(dataset_dir, os.path.basename(file))
                    if not os.path.exists(file_path):
                        print(f"File not found: {file_path}")
                        end = True
                        break
                    else:
                        if file.endswith(".mp4"):
                            basename, video_path, json_path = get_mp4_tuple(
                                file_path, dataset_dir
                            )
                            need_embed[tar_path].append(
                                (label, basename, video_path, json_path)
                            )
            if end:
                del need_embed[tar_path]
                break

    print("number of tar", len(need_embed))
    for tar_name, need_embed_list in need_embed.items():
        results = {name[-2]: [] for name in need_embed_list}
        dataset = BasaltMinecraftDataset(
            dataset_dict=dataset_dict, unique_ids=need_embed_list
        )
        dl = DataLoader(
            dataset,
            batch_size=None,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: x,
            persistent_workers=True,
        )
        batch_size = 32
        last_batch_episode_id = -1 * np.ones(batch_size)
        step_size = 64
        gen = data_generator(
            dl,
            batch_size=batch_size,
            step_size=step_size,
        )
        num_devices = xr.global_runtime_device_count()
        device_ids = np.arange(num_devices)
        mesh_shape = (num_devices,)
        mesh = xs.Mesh(device_ids, mesh_shape, ("data",))
        agent_state = policy.initial_state(batch_size)
        agent_state = tree_map(
            lambda x: (
                x
                if x is None
                else xs.mark_sharding(
                    x.to(agent.device),
                    mesh,
                    (
                        "data",
                        None,
                        None,
                    ),
                )
            ),
            agent_state,
        )
        first = torch.zeros((batch_size, step_size)).bool()
        start_time = time.time()
        for (
            obs,
            actions,
            labels,
            batch_episode_id,
            uids,
        ), mask in gen:
            agent_obs = {
                "img": xs.mark_sharding(
                    obs.to(agent.device), mesh, ("data", None, None, None, None)
                ),
            }

            first[
                (last_batch_episode_id != batch_episode_id) & (batch_episode_id != -1)
            ] = torch.from_numpy(np.array([[True] + [False] * (step_size - 1)]))
            first[
                (last_batch_episode_id == batch_episode_id) & (batch_episode_id != -1)
            ] = False
            last_batch_episode_id = batch_episode_id
            agent_actions = [agent._env_action_to_agent(act) for act in actions]
            with torch_xla.step():
                with torch.no_grad():
                    embedding, new_agent_state = policy.get_output_for_observation(
                        agent_obs,
                        agent_state,
                        first.to(device),
                        return_embedding=True,
                    )
                agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            for i, uid in enumerate(uids):
                if batch_episode_id[i] != -1:
                    results[uid].append((embedding[i].cpu().global_tensor, agent_actions[i], mask[i]))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("FINISH, elapsed_time:", elapsed_time)

        sink = wds.TarWriter(tar_name)
        print(tar_name)
        for subkey, embed in results.items():
            obs = np.concatenate(
                [e[0].numpy()[e[2] == 1] for e in embed]
            )
            actions = {
                key: np.concatenate([e[1][key][e[2] == 1] for e in embed])
                for key in embed[0][1].keys()
            }
            sink.write(
                {
                    "__key__": subkey,
                    "obs.npy": obs,
                    "actions.pyd": actions,
                }
            )
        sink.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
