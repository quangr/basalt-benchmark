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
import pickle
import webdataset as wds

DEVICE = "cuda"


def compute_loss(v, agent_actions, label):
    total_loss = 0

    for key in v.keys():
        pred_v = v[key].squeeze()  # Shape: [batch_size, N]
        targets = torch.tensor(
            [action[key][0] for action in agent_actions], dtype=torch.long
        )  # Shape: [batch_size,1]
        p = pred_v.gather(1, targets.to(pred_v.device))[..., 0].sigmoid()
        loss = label * p.log() + (1 - label) * (1 - p).log()
        total_loss += loss

    return -total_loss.mean()


def calculate_l2_norm_penalty(model):
    l2_norm = 0.0

    for param in model.parameters():
        l2_norm += torch.sum(param**2)

    return l2_norm


@dataclass
class Args:
    in_model: str = "pipeline_test_data/VPT-models/foundation-model-3x.model"
    in_weights: str = "pipeline_test_data/VPT-models/foundation-model-3x.weights"


def train(args: Args):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(args.in_model)

    agent = MineRLAgent(
        device=DEVICE,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(args.in_weights)
    agent.policy.eval()
    policy = agent.policy

    dataset_dict = {
        "downloads/data/demonstrations/MineRLBasaltBuildVillageHouse-v0": 0,
        "downloads/data/demonstrations/MineRLBasaltCreateVillageAnimalPen-v0": 0,
    }

    need_embed_list = []
    new_results = {}
    for dataset_dir, label in dataset_dict.items():
        new_results[os.path.abspath(dataset_dir)] = {}

        unique_ids = glob.glob(os.path.join(dataset_dir, "*.mp4"))

        for uid in unique_ids:
            basename, video_path, json_path = get_mp4_tuple(uid, dataset_dir)
            embed_pickle_path = os.path.abspath(
                os.path.join(dataset_dir, basename + ".pickle")
            )
            if not os.path.exists(embed_pickle_path):
                new_results[os.path.abspath(dataset_dir)][video_path] = []
                need_embed_list.append((label, basename, video_path, json_path))
    print("number of embed demo", len(need_embed_list))
    if len(need_embed_list) > 0:
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
        batch_size = 4
        last_batch_episode_id = -1 * np.ones(batch_size)
        step_size = 64
        gen = data_generator(
            dl,
            batch_size=batch_size,
            step_size=step_size,
        )
        agent_state = policy.initial_state(batch_size)
        first = torch.zeros((batch_size, step_size)).bool().to(DEVICE)
        for (
            obs,
            actions,
            labels,
            batch_episode_id,
            uids,
        ), mask in gen:  # Fetch 3 batches for testing
            agent_obs = {
                "img": obs.to(agent.device),
            }
            if len(last_batch_episode_id) != len(batch_episode_id):
                good_index = [
                    last_batch_episode_id.tolist().index(id)
                    if id in last_batch_episode_id
                    else -1
                    for id in batch_episode_id
                ]  ##TODO what if reset
                agent_state = tree_map(
                    lambda x: x if x is None or good_index == -1 else x[good_index],
                    agent_state,
                )
                first = first[good_index]
                first[good_index != -1] = False
                first[good_index == -1] = torch.from_numpy(
                    np.array([[True] + [False] * (step_size - 1)])
                ).to(DEVICE)
            else:
                if (last_batch_episode_id != batch_episode_id).any():
                    first[last_batch_episode_id != batch_episode_id] = torch.from_numpy(
                        np.array([[True] + [False] * (step_size - 1)])
                    ).to(DEVICE)
                else:
                    first[:] = False
            last_batch_episode_id = batch_episode_id
            agent_actions = [agent._env_action_to_agent(act) for act in actions]
            with torch.no_grad():
                embedding, new_agent_state = agent.policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    first,
                    return_embedding=True,
                )
            for i, uid in enumerate(uids):
                new_results[os.path.dirname(uid)][uid].append(
                    (embedding[i], agent_actions[i], mask[i])
                )
            agent_state = tree_map(lambda x: x.detach(), new_agent_state)

        for key, result_dict in new_results.items():
            sink = wds.TarWriter(os.path.join(key, "webdataset.tar"))

            for subkey, embed in result_dict.items():
                obs = np.concatenate([e[0].cpu().numpy()[e[2] == 1] for e in embed])
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
        # for key, result_dict in new_results.items():
        #     for subkey, embed in result_dict.items():
        #         obs = np.concatenate([e[0].cpu().numpy()[e[2] == 1] for e in embed])
        #         actions = {
        #             key: np.concatenate([e[1][key][e[2] == 1] for e in embed])
        #             for key in embed[0][1].keys()
        #         }
        #         basename = os.path.basename(subkey).split(".")[0]
        #         embed_pickle_path = os.path.abspath(
        #             os.path.join(dataset_dir, basename + ".pickle")
        #         )
        #         with open(embed_pickle_path, "wb") as f:
        #             pickle.dump((obs, actions), f)
        print("FINISH")


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
