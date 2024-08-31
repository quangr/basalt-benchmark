import glob
import torch
import webdataset as wds
from torch.utils.data import DataLoader
import numpy as np
import os
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.config import DefaultDataConfig, TaskType
from basalt.vpt_lib.agent import MineRLAgent
from basalt.adapter import method_dict, FixVPTAdapter
import torch_xla.core.xla_model as xm
import torch_xla
import torch_xla.debug.metrics as met

torch_xla.experimental.eager_mode(True)


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


def process_batches(dataloader, task_dict, batch_size=512):
    for i, batch in enumerate(dataloader):
        task_name = os.path.basename(os.path.dirname(batch[-1]))
        if i == 0:
            batch[-1] = task_dict[task_name] * torch.ones(len(batch[0]))
            buffer = batch
        else:
            batch[-1] = task_dict[task_name] * torch.ones(len(batch[0]))
            buffer = tree_concatenate(buffer, batch)
        length = len(buffer[0])
        new_index = np.random.permutation(length)
        buffer = torch.utils._pytree.tree_map(lambda x: x[new_index], buffer)
        while length >= 10000:
            yield torch.utils._pytree.tree_map(
                lambda x: x[:batch_size].to(device), buffer
            )
            buffer = torch.utils._pytree.tree_map(lambda x: x[batch_size:], buffer)
            length = len(buffer[0])

    while len(buffer[0]) > 0:
        yield torch.utils._pytree.tree_map(lambda x: x[:batch_size].to(device), buffer)
        buffer = torch.utils._pytree.tree_map(lambda x: x[batch_size:], buffer)


@dataclass
class Args:
    data: DefaultDataConfig
    tasks: TaskType = TaskType.CaveVsWater
    method: str = "cls"

    @property
    def dataset_path(self):
        return [f"{self.data.task_data_prefix}/{t}" for t in self.tasks.value]


device = xm.xla_device()
# device = torch.device("cuda")


if __name__ == "__main__":
    args = tyro.cli(Args)
    adapter_dict = method_dict[args.method]
    if adapter_dict["expert_only"]:
        urls = sum(
            [glob.glob(f"{path}/*.tar") for path in [args.dataset_path[1]]],
            [],
        )
    else:
        urls = sum(
            [glob.glob(f"{path}/*.tar") for path in args.dataset_path],
            [],
        )
    print(urls)
    task_dict = {item: index for index, item in enumerate(args.tasks.value)}
    train_dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=True,
        )
        .shuffle(100)
        .decode()
        .to_tuple("mp4.obs.npy", "mp4.actions.pyd", "__url__")
    )
    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=None, num_workers=min(4, len(urls))
    )

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
    policy = agent.policy
    adapter = adapter_dict["adapter"](agent)
    assert isinstance(adapter, FixVPTAdapter)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.0000181)
    import time

    def step_fn(adapter, batch, optimizer):
        optimizer.zero_grad()
        embedding, action, label = batch
        loss = adapter.embed_loss(embedding, action, label)
        loss.backward()
        optimizer.step()
        return loss

    step_fn = torch_xla.experimental.compile(step_fn)

    for epoch in range(50):
        start_time = time.time()
        for batch in process_batches(train_dataloader, task_dict, batch_size=2048):
            loss = step_fn(adapter, batch, optimizer)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("FINISH, elapsed_time:", elapsed_time)
        print(f"Epoch {epoch + 1}, Train Loss: {loss.item()}")

        if (epoch + 1) % 10 == 0:
            save_path = f"checkpoints/{args.method}/epoch_{epoch + 1}.pt"
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            adapter.save_parameters(save_path)
            print(f"Saved chekcpoints at {save_path}")
