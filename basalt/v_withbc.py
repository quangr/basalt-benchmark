# Script for embedding original mp4 trajectories with VPT models into .npz files which contain:
#  1. Embeddings
#  2. Button actions (in VPT action space)
#  3. Camera actions (in VPT action space)
#  4. ESC button (binary)
#  5. Is null action (binary)

from argparse import ArgumentParser
import pickle
import torch.nn as nn

import torch
import numpy as np

from basalt.vpt_lib.agent import MineRLAgent
from basalt.utils.trunk_loader import ContrastMinecraftDataLoader
from basalt.vpt_lib.tree_util import tree_map
from basalt.vpt_lib.cls_head import make_cls_head

DEVICE = "cuda"


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


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
        l2_norm += torch.sum(param ** 2)

    return l2_norm


def embed_trajectories(data_dir, in_model, in_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    agent = MineRLAgent(
        device=DEVICE,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    policy = agent.policy
    cls_head = make_cls_head(
        agent.policy.action_space, agent_policy_kwargs["hidsize"]
    ).to(agent.device)
    cls_head.reset_parameters()
    for param in policy.parameters():
        param.requires_grad = False
    for param in cls_head.parameters():
        param.requires_grad = True

    # Define the optimizer (e.g., Adam optimizer)
    optimizer = torch.optim.Adam(cls_head.parameters(), lr=0.0001)

    last_batch_episode_id = -1
    for epoch in range(100):
        epoch_loss = 0
        num_batches = 0
        data_loader = ContrastMinecraftDataLoader(
            dataset_dir=data_dir,
            batch_size=1,
        )
        for batch_episode_id, batches, label in data_loader:
            agent_obs = {
                "img": torch.from_numpy(np.array([[x[0] for x in batches]])).to(
                    agent.device
                ),
            }
            if last_batch_episode_id != batch_episode_id:
                agent_state = policy.initial_state(1)
                dummy_first = torch.from_numpy(
                    np.array([[True] + [False] * (len(agent_obs["img"][0]) - 1)])
                ).to(DEVICE)
            else:
                dummy_first = torch.from_numpy(
                    np.array([[False] * (len(agent_obs["img"][0]))])
                ).to(DEVICE)
            last_batch_episode_id = batch_episode_id
            agent_actions = [agent._env_action_to_agent(x[1]) for x in batches]

            embedding, new_agent_state = policy.get_output_for_observation(
                agent_obs,
                agent_state,
                dummy_first,
                return_embedding=True,
            )
            v = cls_head(embedding)
            cls_loss = compute_loss(v, agent_actions, label)
            loss=cls_loss+calculate_l2_norm_penalty(cls_head)*1e-2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            agent_state = tree_map(lambda x: x.detach(), new_agent_state)

            epoch_loss += cls_loss.item()
            num_batches += 1

        average_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss}")
        if (epoch + 1) % 10 == 0:
            save_path = f"pipeline_test_data/cls/epoch_{epoch + 1}.pt"
            torch.save(cls_head, save_path)
            print(f"Saved soft_promt at {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0",
        help="Path to the directory containing recordings to be trained on",
    )
    parser.add_argument(
        "--in-model",
        default="pipeline_test_data/VPT-models/foundation-model-3x.model",
        type=str,
        help="Path to the .model file to be finetuned",
    )
    parser.add_argument(
        "--in-weights",
        default="pipeline_test_data/VPT-models/foundation-model-3x.weights",
        type=str,
        help="Path to the .weights file to be finetuned",
    )

    args = parser.parse_args()
    embed_trajectories(
        args.data_dir,
        args.in_model,
        args.in_weights,
    )
