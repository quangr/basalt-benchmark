import time
import torch_xla.runtime as xr
import rich

xr.use_spmd()
import torch
import numpy as np
from dataclasses import dataclass
import tyro
from basalt.common import load_model_parameters
from basalt.vpt_lib.agent import MineRLAgent
from basalt.vpt_lib.tree_util import tree_map
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd.debugging import visualize_tensor_sharding

# torch_xla.experimental.eager_mode(True)
import torch_xla

# device = torch.device("cuda")
device = xm.xla_device()
import torch_xla.core.xla_model as xm
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

    obs = torch.randint(0, 1, size=(64, 64, 128, 128, 3))
    first = torch.zeros((64, 64)).bool()
    agent_state = policy.initial_state(64)
    num_devices = xr.global_runtime_device_count()
    device_ids = np.arange(num_devices)
    mesh_shape = (num_devices,)
    mesh = xs.Mesh(device_ids, mesh_shape, ("data",))
    agent_obs = {
        "img": xs.mark_sharding(
            obs.to(agent.device), mesh, ("data", None, None, None, None)
        ),
    }
    print(visualize_tensor_sharding(agent_obs["img"], use_color=False))
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

    # agent_obs = {
    #     "img": obs.to(agent.device)
    # }
    for i in range(10):
        start_time = time.time()
        with torch_xla.step():
            with torch.no_grad():
                embedding, new_agent_state = policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    first.to(device),
                    return_embedding=True,
                )
            agent_state = tree_map(lambda x: x.detach(), new_agent_state)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"FINISH {i}, elapsed_time:", elapsed_time)


if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)
