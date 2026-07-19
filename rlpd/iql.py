"""IQL agent (offline pretrain + online finetune) implementing the Agent protocol."""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .interfaces import Batch
from .networks import Actor, EnsembleCritic, ValueNet


def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.where(diff > 0, tau, 1.0 - tau) * diff.pow(2)


class IQL:
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
        self.device = device
        algo, iql = cfg["algo"], cfg["iql"]
        width = int(algo["hidden_width"])
        num_layers = int(cfg.get("env_specific", {}).get("num_layers", 2))
        lr = float(algo["learning_rate"])
        self.gamma = float(algo["gamma"])
        self.tau = float(iql["tau"])
        self.expectile = float(iql["expectile"])
        self.beta = float(iql["beta"])
        self.adv_clip = float(iql["adv_clip"])

        self.actor = Actor(obs_dim, act_dim, width, num_layers, log_std_min=-5.0).to(device)
        self.critic = EnsembleCritic(obs_dim, act_dim, width, num_layers, 2, layernorm=False).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.value = ValueNet(obs_dim, width, num_layers).to(device)
        self._c = list(self.critic.parameters())
        self._ct = list(self.critic_target.parameters())

        fused = torch.device(device).type == "cuda"
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, fused=fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, fused=fused)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr, fused=fused)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch: Batch, update_actor: bool = True) -> Dict[str, float]:
        obs, act, next_obs = batch["obs"], batch["action"], batch["next_obs"]

        # value: expectile regression toward min target-Q
        with torch.no_grad():
            q_t = self.critic_target(obs, act).min(dim=0).values
        v = self.value(obs)
        v_loss = _expectile_loss(q_t - v, self.expectile).mean()
        self.value_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        self.value_opt.step()

        # critic: TD with V(s') as bootstrap
        with torch.no_grad():
            y = batch["reward"] + self.gamma * (1.0 - batch["done"]) * self.value(next_obs)
        q = self.critic(obs, act)
        q_loss = F.mse_loss(q, y.unsqueeze(0).expand_as(q))
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.critic_opt.step()

        # policy: advantage-weighted regression onto batch actions
        with torch.no_grad():
            weight = torch.exp(self.beta * (q_t - v)).clamp(max=self.adv_clip)
        actor_loss = -(weight * self.actor.log_prob(obs, act)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            torch._foreach_mul_(self._ct, 1.0 - self.tau)
            torch._foreach_add_(self._ct, self._c, alpha=self.tau)

        if not update_actor:
            return {}
        return {
            "value_loss": float(v_loss.item()),
            "critic_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "mean_q": float(q.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "value": self.value.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
        }

    def load_state_dict(self, s: dict) -> None:
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.critic_target.load_state_dict(s["critic_target"])
        self.value.load_state_dict(s["value"])
        self.actor_opt.load_state_dict(copy.deepcopy(s["actor_opt"]))
        self.critic_opt.load_state_dict(copy.deepcopy(s["critic_opt"]))
        self.value_opt.load_state_dict(copy.deepcopy(s["value_opt"]))
