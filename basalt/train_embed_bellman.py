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
import torch_xla.core.xla_model as xm
import torch_xla


def identity(x):
    return x


def tree_concatenate(tree1, tree2):
    if isinstance(tree1, dict) and isinstance(tree2, dict):
        return {k: tree_concatenate(tree1[k], tree2[k]) for k in tree1.keys()}
    elif isinstance(tree1, list) and isinstance(tree2, list):
        return [tree_concatenate(t1, t2) for t1, t2 in zip(tree1, tree2)]
    elif isinstance(tree1, tuple) and isinstance(tree2, tuple):
        return tuple(tree_concatenate(t1, t2) for t1, t2 in zip(tree1, tree2))
    else:
        return torch.concatenate([tree1, tree2])


def process_batches(dataloader, batch_size=512):
    for i, batch in enumerate(dataloader):
        batch = [
            *torch.utils._pytree.tree_map(lambda x: x[:-1], batch),
            batch[0][1:],
        ]
        if i == 0:
            batch[2] = ("MineRLBasaltMakeWaterfall" in batch[2]) * torch.ones(
                len(batch[0])
            )
            buffer=batch
        else:
            batch[2] = ("MineRLBasaltMakeWaterfall" in batch[2]) * torch.ones(
                len(batch[0])
            )
            buffer = tree_concatenate(buffer, batch)
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
    in_model: str = "/data/foundation-model-3x.model"
    in_weights: str = "/data/foundation-model-3x.weights"
    method: str = "cls_bellman"


device = xm.xla_device()


if __name__ == "__main__":
    args = tyro.cli(Args)
    train_dataset = (
        wds.WebDataset(
            wds.shardlists.expand_urls(
                "/data/demonstrations/MineRLBasaltMakeWaterfall-v0/{1..7}.tar"
            )
            + wds.shardlists.expand_urls(
                "/data/demonstrations/MineRLBasaltFindCave-v0/{0..7}.tar"
            ),
            shardshuffle=True,
        )
        .shuffle(100)
        .decode()
        .to_tuple("mp4.obs.npy", "mp4.actions.pyd", "__url__")
    )
    train_dataloader = wds.WebLoader(train_dataset, batch_size=None, num_workers=4)

    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(args.in_model)

    agent = MineRLAgent(
        device=device,
        policy_kwargs=agent_policy_kwargs,
        pi_head_kwargs=agent_pi_head_kwargs,
    )
    agent.load_weights(args.in_weights)
    agent.policy.eval()
    policy = agent.policy
    adapter_dict = method_dict[args.method]
    adapter = adapter_dict["adapter"](agent)
    assert isinstance(adapter, FixVPTAdapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=3e-4)
    import time

    for epoch in range(50):
        epoch_loss = 0
        num_batches = 0
        start_time = time.time()
        step = 0
        for batch in process_batches(train_dataloader, batch_size=2048):
            with torch_xla.step():
                optimizer.zero_grad()
                batch = torch.utils._pytree.tree_map(lambda x: x.to(device), batch)
                embedding, action, label, next_embedding = batch
                loss = adapter.embed_loss(embedding, action, label, next_embedding)
                loss.backward()
                # optimizer.step()
                xm.optimizer_step(optimizer)

                epoch_loss += loss.detach()
                num_batches += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("FINISH, elapsed_time:", elapsed_time)
        average_train_loss = epoch_loss.item() / num_batches
        print(f"Epoch {epoch + 1}, Average Train Loss: {average_train_loss}")

        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoints/{args.method}/epoch_{epoch + 1}.pt"
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            adapter.save_parameters(save_path)
            print(f"Saved chekcpoints at {save_path}")
