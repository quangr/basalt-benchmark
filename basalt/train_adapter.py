import os
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from basalt.utils.trunk_loader import MinecraftDataLoader
from basalt.vpt_lib.tree_util import tree_map
from basalt.adapter import method_dict

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

    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.005)

    last_batch_episode_id = -1
    for epoch in range(500):
        epoch_loss = 0
        num_batches = 0
        data_loader = MinecraftDataLoader(
            dataset_dir=args.data_dir, batch_size=1, contrast=adapter_dict["contrast_dataset"]
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

            loss, new_agent_state = adapter.loss(
                agent_obs, agent_state, dummy_first, agent_actions, label
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            agent_state = tree_map(lambda x: x.detach(), new_agent_state)

            epoch_loss += loss.item()
            num_batches += 1

        average_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}, Average Loss: {average_epoch_loss}")
        if (epoch + 1) % 100 == 0:
            save_path = f"checkpoints/{args.method}/epoch_{epoch + 1}.pt"
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            adapter.save_parameters(save_path)
            print(f"Saved chekcpoints at {save_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
