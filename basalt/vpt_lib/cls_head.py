import logging
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from gym3.types import DictType, Discrete, Real, TensorType, ValType

LOG0 = -100


def fan_in_linear(module: nn.Module, scale=1.0, bias=True):
    """Fan-in init"""
    module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

    if bias:
        module.bias.data *= 0


class Clshead(nn.Module):
    """Abstract base class for action heads compatible with forc"""

    def forward(self, input_data: torch.Tensor) -> Any:
        """
        Just a forward pass through this head
        :returns pd_params - parameters describing the probability distribution
        """
        raise NotImplementedError



class CategoricalClshead(Clshead):
    """Action head with categorical actions"""

    def __init__(
        self, input_dim: int, shape: Tuple[int], num_actions: int, builtin_linear_layer: bool = True, temperature: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.output_shape = shape + (num_actions,)
        self.temperature = temperature

        if builtin_linear_layer:
            # self.linear_layer = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, np.prod(self.output_shape)))
            self.linear_layer =  nn.Sequential(nn.Linear(input_dim, np.prod(self.output_shape)))
        else:
            assert (
                input_dim == num_actions
            ), f"If input_dim ({input_dim}) != num_actions ({num_actions}), you need a linear layer to convert them."
            self.linear_layer = None

    def reset_parameters(self):
        if self.linear_layer is not None:
            for linear_layer in self.linear_layer[:-1]:
                if isinstance(linear_layer, nn.Linear):
                    init.kaiming_normal_(linear_layer.weight)

            init.orthogonal_(self.linear_layer[-1].weight, gain=0.01)
            init.constant_(self.linear_layer[-1].bias, 0.0)
            fan_in_linear(self.linear_layer[-1], scale=0.01)

    def forward(self, input_data: torch.Tensor, mask=None) -> Any:
        if self.linear_layer is not None:
            flat_out = self.linear_layer(input_data)
        else:
            flat_out = input_data
        shaped_out = flat_out.reshape(flat_out.shape[:-1] + self.output_shape)
        return shaped_out.float()


class DictClshead(nn.ModuleDict):
    """Action head with multiple sub-actions"""

    def reset_parameters(self):
        for subhead in self.values():
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        """
        :param kwargs: each kwarg should be a dict with keys corresponding to self.keys()
                e.g. if this ModuleDict has submodules keyed by 'A', 'B', and 'C', we could call:
                    forward(input_data, foo={'A': True, 'C': False}, bar={'A': 7}}
                Then children will be called with:
                    A: forward(input_data, foo=True, bar=7)
                    B: forward(input_data)
                    C: forward(input_Data, foo=False)
        """
        result = {}
        for head_name, subhead in self.items():
            head_kwargs = {
                kwarg_name: kwarg[head_name]
                for kwarg_name, kwarg in kwargs.items()
                if kwarg is not None and head_name in kwarg
            }
            result[head_name] = subhead(input_data, **head_kwargs)
        return result

    def logprob(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return sum(subhead.logprob(actions[k], logits[k]) for k, subhead in self.items())

    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> Any:
        return {k: subhead.sample(logits[k], deterministic) for k, subhead in self.items()}

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        return sum(subhead.entropy(logits[k]) for k, subhead in self.items())

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        return sum(subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in self.items())


def make_cls_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0):
    """Helper function to create an action head corresponding to the environment action space"""
    if isinstance(ac_space, TensorType):
        if isinstance(ac_space.eltype, Discrete):
            return CategoricalClshead(pi_out_size, ac_space.shape, ac_space.eltype.n, temperature=temperature)
        elif isinstance(ac_space.eltype, Real):
            raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")
    elif isinstance(ac_space, DictType):
        return DictClshead({k: make_cls_head(v, pi_out_size, temperature) for k, v in ac_space.items()})
    raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")
