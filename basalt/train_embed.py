import torch
import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import os
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from basalt.adapter import method_dict, FixVPTAdapter


def identity(x):
    return x


def process_batches(dataloader, batch_size=512):
    for i, batch in enumerate(dataloader):
        if i == 0:
            batch[-1] = ("MineRLBasaltBuildVillageHouse" in batch[-1]) * torch.ones(
                len(batch[0])
            )
            buffer = batch
        else:
            batch[-1] = ("MineRLBasaltBuildVillageHouse" in batch[-1]) * torch.ones(
                len(batch[0])
            )
            buffer = torch.utils._pytree.tree_map(
                lambda x, y: np.concatenate([x, y]), buffer, batch
            )
        length = len(buffer[0])
        new_index = np.random.permutation(length)
        buffer = torch.utils._pytree.tree_map(lambda x: x[new_index], buffer)
        while length >= 10000:
            yield torch.utils._pytree.tree_map(lambda x: x[:batch_size], buffer)
            buffer = torch.utils._pytree.tree_map(lambda x: x[batch_size:], buffer)
            length = len(buffer[0])

    while len(buffer[0]) > 0:
        yield torch.utils._pytree.tree_map(lambda x: x[:batch_size], buffer)
        buffer = torch.utils._pytree.tree_map(lambda x: x[batch_size:], buffer)


@dataclass
class Args:
    in_model: str = "pipeline_test_data/VPT-models/foundation-model-3x.model"
    in_weights: str = "pipeline_test_data/VPT-models/foundation-model-3x.weights"
    method: str = "cls"


DEVICE = "cuda"

if __name__ == "__main__":
    args = tyro.cli(Args)
    dataset = (
        wds.WebDataset(
            [
                "downloads/data/demonstrations/MineRLBasaltBuildVillageHouse-v0/webdataset.tar",
                "downloads/data/demonstrations/MineRLBasaltCreateVillageAnimalPen-v0/webdataset.tar",
            ]
        )
        .shuffle(100)  # Shuffle the dataset with a buffer size of 100
        .decode()  # Decode the data
        .to_tuple("mp4.obs.npy", "mp4.actions.pyd", "__url__")  # Select specific fields
    )

    # Create a DataLoader to fetch data in batches of size 4
    dataloader = DataLoader(dataset, batch_size=None)

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
    assert isinstance(adapter, FixVPTAdapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.0001)

    for epoch in range(50):
        epoch_loss = 0
        num_batches = 0
        for batch in process_batches(dataloader, batch_size=512):
            batch = torch.utils._pytree.tree_map(
                lambda x: torch.tensor(x).to(DEVICE), batch
            )
            embedding, action, label = batch
            loss = adapter.embed_loss(embedding, action, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
