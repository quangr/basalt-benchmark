import os
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from basalt.utils.chunk_loader import MinecraftDataset, data_generator
from basalt.vpt_lib.tree_util import tree_map
from basalt.adapter import method_dict
from torch.utils.data import DataLoader

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
    data_dir: str = "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0"
    in_model: str = "pipeline_test_data/VPT-models/foundation-model-3x.model"
    in_weights: str = "pipeline_test_data/VPT-models/foundation-model-3x.weights"
    method: str = "soft_adapter"


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
    adapter_dict = method_dict[args.method]
    adapter = adapter_dict["adapter"](agent)

    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
    dataset_dict = {
        "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0": 0,
        # "pipeline_test_data/demonstrations/MineRLBasaltMakeWaterfall-v0-fail": 1,
    }
    dataset = MinecraftDataset(dataset_dict=dataset_dict, contrast=False)
    dl = DataLoader(
        dataset,
        batch_size=None,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x,
        persistent_workers=True,
    )
    for epoch in range(500):
        epoch_loss = 0
        num_batches = 0

        batch_size = 2
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

            loss, new_agent_state = adapter.loss(
                agent_obs, mask, agent_state, first, agent_actions, labels
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            agent_state = tree_map(lambda x: x.detach(), new_agent_state)

            epoch_loss += loss.item()
            num_batches += 1

        average_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss}")
        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoints/{args.method}/epoch_{epoch + 1}.pt"
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            adapter.save_parameters(save_path)
            print(f"Saved chekcpoints at {save_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
