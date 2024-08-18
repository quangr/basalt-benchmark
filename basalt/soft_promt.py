# Script for embedding original mp4 trajectories with VPT models into .npz files which contain:
#  1. Embeddings
#  2. Button actions (in VPT action space)
#  3. Camera actions (in VPT action space)
#  4. ESC button (binary)
#  5. Is null action (binary)

from argparse import ArgumentParser
import pickle
import time
import os
import torch.nn as nn

import torch
import numpy as np
import tqdm

from basalt.vpt_lib.agent import MineRLAgent
from basalt.utils.trunk_loader import MinecraftDataLoader
from basalt.vpt_lib.tree_util import tree_map

DEVICE = "cuda"


def load_model_parameters(path_to_model_file):
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def compute_cross_entropy_loss(policy_output, agent_actions):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    for key in policy_output.keys():
        pred_logits = policy_output[key].squeeze()  # Shape: [batch_size, N]
        targets = torch.tensor(
            [action[key][0][0] for action in agent_actions], dtype=torch.long
        )

        loss = criterion(pred_logits, targets.to(pred_logits.device))
        total_loss += loss

    return total_loss


def embed_trajectories(data_dir, in_model, in_weights):
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)

    agent = MineRLAgent(
        device=DEVICE,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(in_weights)
    agent.policy.eval()

    policy = agent.policy
    for param in policy.parameters():
        param.requires_grad = False

    soft_promt = nn.Parameter(torch.zeros(3072,device=agent.device))

    # Define the optimizer (e.g., Adam optimizer)
    optimizer = torch.optim.Adam([soft_promt], lr=0.001)

    last_batch_episode_id = -1
    for epoch in range(1000):
        epoch_loss = 0
        num_batches = 0
        data_loader = MinecraftDataLoader(
            dataset_dir=data_dir,
            batch_size=1,
        )
        for batch_episode_id, batches in data_loader:
            agent_obs = {
                "img": torch.from_numpy(np.array([[x[0] for x in batches]])).to(
                    agent.device
                ),
                "soft_promt": soft_promt,
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
            pi_logits = policy.pi_head(embedding)
            # loss = compute_cross_entropy_loss(pi_logits, agent_actions)
            log_loss=-policy.pi_head.logprob({key: torch.tensor(np.array([x[key][0] for x in agent_actions])).to(agent.device) for key in agent_actions[0].keys()}, pi_logits).mean()
            loss=log_loss+(soft_promt.abs()).sum()/50
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            
            epoch_loss += log_loss.item()
            num_batches += 1

        average_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss}")
        if (epoch + 1) % 100 == 0:
            save_path = f"pipeline_test_data/soft_promt/epoch_{epoch + 1}.pt"
            torch.save(soft_promt, save_path)
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
