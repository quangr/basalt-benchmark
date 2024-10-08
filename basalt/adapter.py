from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
from basalt.vpt_lib.tree_util import tree_map
from basalt.vpt_lib.agent import MineRLAgent
from basalt.vpt_lib.cls_head import make_cls_head


class AbstractAdapter(ABC):
    def __init__(self, vpt_agent):
        self.vpt_agent = vpt_agent
        for param in self.vpt_agent.policy.parameters():
            param.requires_grad = False

    @abstractmethod
    def load_parameters(self, weight_path):
        pass

    @abstractmethod
    def save_parameters(self, weight_path):
        pass

    @abstractmethod
    def loss(self, agent_obs, mask, agent_state, first, agent_actions, label):
        pass

    @abstractmethod
    def compute_action(self, agent_obs, agent_state, first):
        pass


class FixVPTAdapter(AbstractAdapter, ABC):
    def loss(self, agent_obs, mask, agent_state, first, agent_actions, label):
        embedding, new_agent_state = self.vpt_agent.policy.get_output_for_observation(
            agent_obs,
            agent_state,
            first,
            return_embedding=True,
        )
        loss = self.embed_loss(embedding, agent_actions, label)
        return loss, new_agent_state

    @abstractmethod
    def embed_loss(self, embedding, action, label):
        pass


class BCAdapter(nn.Module, FixVPTAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.head = self.vpt_agent.policy.pi_head
        self.head.reset_parameters()
        for param in self.head.parameters():
            param.requires_grad = True

    def load_parameters(self, weight_path):
        self.head.load_state_dict(torch.load(weight_path))

    def save_parameters(self, weight_path):
        torch.save(self.head.state_dict(), weight_path)

    def embed_loss(self, embedding, action, label):
        pi_logits = self.head(embedding)
        log_loss = -self.head.logprob(action, pi_logits).mean()
        return log_loss

    def compute_action(self, agent_obs, agent_state, first):
        with torch.no_grad():
            embedding, new_agent_state = (
                self.vpt_agent.policy.get_output_for_observation(
                    {"img": agent_obs["img"]},
                    agent_state,
                    first,
                    return_embedding=True,
                )
            )
            pd = self.head(embedding)
            input_pd = pd
            ac = self.head.sample(input_pd, deterministic=False)
            ac = tree_map(lambda x: x[:, 0], ac)

            minerl_action = self.vpt_agent._agent_action_to_env(ac)
            minerl_action["ESC"] = np.zeros_like(minerl_action["attack"])
        return minerl_action, new_agent_state


class FineTuneBCAdapter(BCAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.head = self.vpt_agent.policy.pi_head
        for param in self.head.parameters():
            param.requires_grad = True


class CLSAdapter(nn.Module, FixVPTAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.head = make_cls_head(
            vpt_agent.policy.action_space, vpt_agent.policy.net.hidsize
        ).to(vpt_agent.device)
        self.head.reset_parameters()

    def load_parameters(self, weight_path):
        self.head.load_state_dict(torch.load(weight_path))

    def save_parameters(self, weight_path):
        torch.save(self.head.state_dict(), weight_path)

    def embed_loss(self, embedding, action, label):
        v = self.head(embedding)
        cls_loss = self.compute_loss(v, action, label)
        return cls_loss

    def compute_loss(self, v, action, label):
        total_loss = 0

        for key in v.keys():
            pred_v = v[key].squeeze()  # Shape: [batch_size, N]
            targets = action[key]  # Shape: [batch_size,1]
            p = pred_v.gather(1, targets.to(pred_v.device)).squeeze().sigmoid()
            loss = (
                label * p.clamp(min=1e-8).log()
                + (1 - label) * (1 - p).clamp(min=1e-8).log()
            )
            total_loss += loss

        return -total_loss.mean()

    def embed_logit(self, embedding, action, label, w=1):
        pd = self.vpt_agent.policy.pi_head(embedding)
        v = self.head(embedding)
        input_pd = {
            key: pd[key] + w * torch.where(label, 1, -1)[:, None, None] * v[key]
            for key in v.keys()
        }
        log_loss = -self.vpt_agent.policy.pi_head.logprob(action, input_pd).mean()

        return log_loss

    def compute_action(self, agent_obs, agent_state, first, w=1):
        with torch.no_grad():
            embedding, new_agent_state = (
                self.vpt_agent.policy.get_output_for_observation(
                    {"img": agent_obs["img"]},
                    agent_state,
                    first,
                    return_embedding=True,
                )
            )
            pd = self.vpt_agent.policy.pi_head(embedding)
            v = self.head(embedding)
            input_pd = {key: pd[key] - w * v[key] for key in v.keys()}
            # input_pd = {key: pd[key] + w * v[key].sigmoid().log() for key in v.keys()}
            ac = self.vpt_agent.policy.pi_head.sample(input_pd, deterministic=False)
            ac = tree_map(lambda x: x[:, 0], ac)

            minerl_action = self.vpt_agent._agent_action_to_env(ac)
            minerl_action["ESC"] = np.zeros_like(minerl_action["attack"])
        return minerl_action, new_agent_state


class SFTCLSAdapter(nn.Module, FixVPTAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.head = make_cls_head(
            vpt_agent.policy.action_space, vpt_agent.policy.net.hidsize
        ).to(vpt_agent.device)
        self.head.reset_parameters()

        self.STF_head = self.vpt_agent.policy.pi_head
        self.STF_head.reset_parameters()
        for param in self.STF_head.parameters():
            param.requires_grad = True

    def load_parameters(self, weight_path):
        head_dict, STF_head_dict = torch.load(weight_path)
        self.head.load_state_dict(head_dict)
        self.STF_head.load_state_dict(STF_head_dict)

    def save_parameters(self, weight_path):
        torch.save((self.head.state_dict(), self.STF_head.state_dict()), weight_path)

    def embed_loss(self, embedding, action, label):
        v = self.head(embedding)
        cls_loss = self.compute_loss(v, action, label)

        pi_logits = self.STF_head(embedding)
        log_loss = -self.STF_head.logprob(action, pi_logits).mean()

        return cls_loss + log_loss

    def compute_loss(self, v, action, label):
        total_loss = 0

        for key in v.keys():
            pred_v = v[key].squeeze()  # Shape: [batch_size, N]
            targets = action[key]  # Shape: [batch_size,1]
            p = pred_v.gather(1, targets.to(pred_v.device)).squeeze().sigmoid()
            loss = (
                label * p.clamp(min=1e-8).log()
                + (1 - label) * (1 - p).clamp(min=1e-8).log()
            )
            total_loss += loss

        return -total_loss.mean()

    def embed_logit(self, embedding, action, label, w=1):
        pd = self.vpt_agent.policy.pi_head(embedding)
        v = self.head(embedding)
        input_pd = {
            key: pd[key] + w * torch.where(label, 1, -1)[:, None, None] * v[key]
            for key in v.keys()
        }
        log_loss = -self.vpt_agent.policy.pi_head.logprob(action, input_pd).mean()

        return log_loss

    def compute_action(self, agent_obs, agent_state, first, w=1):
        with torch.no_grad():
            embedding, new_agent_state = (
                self.vpt_agent.policy.get_output_for_observation(
                    {"img": agent_obs["img"]},
                    agent_state,
                    first,
                    return_embedding=True,
                )
            )
            pd = self.STF_head(embedding)
            v = self.head(embedding)
            input_pd = {key: pd[key] - w * v[key] for key in v.keys()}
            # input_pd = {key: pd[key] + w * v[key].sigmoid().log() for key in v.keys()}
            ac = self.STF_head.sample(input_pd, deterministic=False)
            ac = tree_map(lambda x: x[:, 0], ac)

            minerl_action = self.vpt_agent._agent_action_to_env(ac)
            minerl_action["ESC"] = np.zeros_like(minerl_action["attack"])
        return minerl_action, new_agent_state


class CLSBellmanAdapter(nn.Module, FixVPTAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.head = make_cls_head(
            vpt_agent.policy.action_space, vpt_agent.policy.net.hidsize
        ).to(vpt_agent.device)
        self.head.reset_parameters()

    def load_parameters(self, weight_path):
        self.head.load_state_dict(torch.load(weight_path))

    def save_parameters(self, weight_path):
        torch.save(self.head.state_dict(), weight_path)

    def embed_loss(self, embedding, action, label, next_embedding):
        v = self.head(embedding)
        next_v = self.head(next_embedding)
        next_logit = self.vpt_agent.policy.pi_head(next_embedding)

        neg_r = -1
        gamma = 0.99
        alpha_coeff = (1 - gamma) * 1e-4

        total_loss = 0

        for key in v.keys():
            pred_v = v[key].squeeze()  # Shape: [batch_size, N]
            next_pred_v = next_v[key].squeeze()  # Shape: [batch_size, N]
            targets = action[key]  # Shape: [batch_size,1]
            policy = next_logit[key].squeeze()
            pred_v = pred_v.gather(1, targets.to(pred_v.device)).squeeze()
            with torch.no_grad():
                v_target = (
                    gamma
                    * (next_pred_v * torch.softmax(next_pred_v + policy, -1)).sum(-1)
                    + neg_r
                )
            loss = (
                (label * (pred_v + neg_r / (1 - gamma)) ** 2).mean()
                + alpha_coeff * ((1 - label) * (pred_v) ** 2).mean()
                + 0.5 * (pred_v - v_target) ** 2
            )
            total_loss += loss
        cls_loss = total_loss.mean()
        return cls_loss

    def embed_logit(self, embedding, action, label, w=1):
        pd = self.vpt_agent.policy.pi_head(embedding)
        v = self.head(embedding)
        input_pd = {
            key: pd[key] + w * torch.where(label, 1, -1)[:, None, None] * v[key]
            for key in v.keys()
        }
        log_loss = -self.vpt_agent.policy.pi_head.logprob(action, input_pd).mean()

        return log_loss

    def compute_action(self, agent_obs, agent_state, first, w=1):
        with torch.no_grad():
            embedding, new_agent_state = (
                self.vpt_agent.policy.get_output_for_observation(
                    {"img": agent_obs["img"]},
                    agent_state,
                    first,
                    return_embedding=True,
                )
            )
            pd = self.vpt_agent.policy.pi_head(embedding)
            v = self.head(embedding)
            input_pd = {key: pd[key] + w * v[key] for key in v.keys()}
            # input_pd = {key: pd[key] + w * v[key].sigmoid().log() for key in v.keys()}
            ac = self.vpt_agent.policy.pi_head.sample(input_pd, deterministic=False)
            ac = tree_map(lambda x: x[:, 0], ac)

            minerl_action = self.vpt_agent._agent_action_to_env(ac)
            minerl_action["ESC"] = np.zeros_like(minerl_action["attack"])
        return minerl_action, new_agent_state


class NoisyCLSAdapter(CLSAdapter):

    def compute_loss(self, v, action, label):
        total_loss = 0

        for key in v.keys():
            pred_v = v[key].squeeze()  # Shape: [batch_size, N]
            targets = action[key]  # Shape: [batch_size,1]
            p = pred_v.gather(1, targets.to(pred_v.device)).squeeze().sigmoid()
            loss = (
                label * p.clamp(min=1e-8).log()
                + (1 - label) * (1 - p).clamp(min=1e-8).log()
            )
            total_loss += loss

        return -total_loss.mean()


class SoftPromptAdapter(nn.Module, AbstractAdapter):
    def __init__(self, vpt_agent):
        nn.Module.__init__(self)
        AbstractAdapter.__init__(self, vpt_agent)
        self.soft_promt = nn.Parameter(
            torch.zeros(vpt_agent.policy.net.hidsize, device=vpt_agent.device)
        )

    def load_parameters(self, weight_path):
        self.soft_promt = torch.load(weight_path)

    def save_parameters(self, weight_path):
        torch.save(self.soft_promt, weight_path)

    def loss(self, agent_obs, mask, agent_state, first, agent_actions, label):
        agent_obs["soft_promt"] = self.soft_promt
        embedding, new_agent_state = self.vpt_agent.policy.get_output_for_observation(
            agent_obs,
            agent_state,
            first,
            return_embedding=True,
        )
        pi_logits = self.vpt_agent.policy.pi_head(embedding)
        log_loss = -self.vpt_agent.policy.pi_head.logprob(
            {
                key: torch.tensor(np.array([x[key] for x in agent_actions])).to(
                    self.vpt_agent.device
                )
                for key in agent_actions[0].keys()
            },
            pi_logits,
        )
        mask = torch.tensor(mask).to(log_loss.device)
        log_loss = (log_loss * mask).sum() / mask.sum()
        return log_loss, new_agent_state

    def compute_action(self, agent_obs, agent_state, first):
        with torch.no_grad():
            agent_obs["soft_promt"] = self.soft_promt
            embedding, new_agent_state = (
                self.vpt_agent.policy.get_output_for_observation(
                    {"img": agent_obs["img"]},
                    agent_state,
                    first,
                    return_embedding=True,
                )
            )
            pd = self.vpt_agent.policy.pi_head(embedding)
            ac = self.vpt_agent.policy.pi_head.sample(pd, deterministic=False)
            ac = tree_map(lambda x: x[:, 0], ac)

            minerl_action = self.vpt_agent._agent_action_to_env(ac)
            minerl_action["ESC"] = np.zeros_like(minerl_action["attack"])
        return minerl_action, new_agent_state


method_dict = {
    "BC": {"adapter": BCAdapter, "expert_only": True},
    "BC_finetune": {"adapter": FineTuneBCAdapter, "expert_only": True},
    "cls": {"adapter": CLSAdapter, "expert_only": False},
    "cls_bellman": {"adapter": CLSBellmanAdapter, "expert_only": False},
    "cls_no_constrast": {"adapter": CLSAdapter, "expert_only": False},
    "soft_adapter": {"adapter": SoftPromptAdapter, "expert_only": False},
    "sft_cls": {"adapter": SFTCLSAdapter, "expert_only": False},
}
# import pickle
# def load_model_parameters(path_to_model_file):
#     agent_parameters = pickle.load(open(path_to_model_file, "rb"))
#     policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
#     pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
#     pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
#     return policy_kwargs, pi_head_kwargs

# if __name__ == "__main__":
#     agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters("downloads/data/VPT-models/foundation-model-1x.model")

#     vpt_agent = MineRLAgent(
#         device="cuda",
#         policy_kwargs=agent_policy_kwargs,
#         pi_head_kwargs=agent_pi_head_kwargs,
#     )
#     vpt_agent.load_weights("downloads/data/VPT-models/foundation-model-1x.weights")
#     vpt_agent.policy.eval()
#     adapter = BCAdapter(vpt_agent)
#     assert list(adapter.parameters())==list(adapter.head.parameters())
