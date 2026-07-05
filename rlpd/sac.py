"""RLPD/SAC agent implementing the Agent protocol."""

from __future__ import annotations

import copy
import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .interfaces import Batch
from .networks import Actor, EnsembleCritic


class RLPDAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: dict, device: str = "cuda"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = device
        self.cfg = cfg

        algo = cfg["algo"]
        env_specific = cfg.get("env_specific", {})

        # --- algorithm hyperparameters ---
        self.gamma = float(algo["gamma"])
        self.tau = float(algo["critic_ema_rho"])  # target-network soft-update rate
        self.ensemble_size = int(algo["ensemble_size"])
        self.cdq = bool(env_specific.get("cdq", True))
        self.entropy_backups = bool(env_specific.get("entropy_backups", True))
        # Size of the random target subset Z: 2 critics (min over them) for
        # clipped double-Q, otherwise a single random critic.
        self.subset_size = 2 if self.cdq else 1

        # --- networks ---
        lr = float(algo["learning_rate"])
        width = int(algo["hidden_width"])
        num_layers = int(env_specific.get("num_layers", 2))
        layernorm = bool(algo["layernorm"])

        self.actor = Actor(obs_dim, act_dim, width, num_layers).to(device)
        self.critic = EnsembleCritic(
            obs_dim, act_dim, width, num_layers, self.ensemble_size, layernorm
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        # Cached param lists so the EMA update is two batched foreach kernels.
        self._critic_params = list(self.critic.parameters())
        self._target_params = list(self.critic_target.parameters())

        # --- entropy temperature (learned in log-space so alpha stays positive) ---
        self.target_entropy = float(algo["target_entropy_scale"]) * act_dim
        self.log_alpha = torch.tensor(
            math.log(float(algo["init_entropy_temp"])),
            device=device,
            requires_grad=True,
        )

        # --- optimizers ---
        # Fused Adam runs each optimizer step as one kernel — significant at UTD=20.
        fused = torch.device(device).type == "cuda"
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr, fused=fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, fused=fused)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr, fused=fused)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict:
        """Full training state (networks + optimizers) for checkpoint/resume."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        with torch.no_grad():  # copy_ keeps the tensor the alpha optimizer holds
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        # deepcopy: Optimizer.load_state_dict aliases input tensors whose
        # dtype/device already match, which would share state with the source
        # agent when loading from a live state_dict (fine from torch.load).
        self.actor_opt.load_state_dict(copy.deepcopy(state["actor_opt"]))
        self.critic_opt.load_state_dict(copy.deepcopy(state["critic_opt"]))
        self.alpha_opt.load_state_dict(copy.deepcopy(state["alpha_opt"]))

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ update

    def update(self, batch: Batch, update_actor: bool = True) -> Dict[str, float]:
        """One gradient step (Algorithm 1). Returns metrics only when
        `update_actor` is True — pulling scalars off the GPU forces a sync, so
        the critic-only calls in the UTD inner loop skip it entirely."""
        obs = batch["obs"]
        action = batch["action"]
        alpha = self.alpha

        y = self._critic_target_values(batch, alpha)
        critic_loss, q_pred = self._update_critic(obs, action, y)

        metrics: Dict[str, float] = {}
        if update_actor:
            metrics = {
                "critic_loss": float(critic_loss.item()),
                "alpha": float(alpha.item()),
                "mean_q": float(q_pred.mean().item()),
            }
            metrics.update(self._update_actor_and_temperature(obs, alpha))

        self._soft_update_target()
        return metrics

    def _critic_target_values(self, batch: Batch, alpha: torch.Tensor) -> torch.Tensor:
        """the TD target y = r + gamma * (1 - done) * Q_target(s', a')."""
        with torch.no_grad():
            next_action, next_logp = self.actor(batch["next_obs"])
            target_q_all = self.critic_target(
                batch["next_obs"], next_action
            )  # (E, N, 1)

            # Random subset Z of the target ensemble; min over Z (clipped double-Q).
            idx = torch.randperm(self.ensemble_size, device=self.device)[
                : self.subset_size
            ]
            q_next = target_q_all[idx].min(dim=0).values  # (N, 1)

            if self.entropy_backups:
                q_next = q_next - alpha * next_logp

            not_done = 1.0 - batch["done"]
            return batch["reward"] + self.gamma * not_done * q_next  # (N, 1)

    def _update_critic(
        self, obs: torch.Tensor, action: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step 2: every ensemble member regresses to the same target y."""
        q_pred = self.critic(obs, action)  # (E, N, 1)
        critic_loss = F.mse_loss(q_pred, y.unsqueeze(0).expand_as(q_pred))

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return critic_loss, q_pred

    def _update_actor_and_temperature(
        self, obs: torch.Tensor, alpha: torch.Tensor
    ) -> Dict[str, float]:
        """Steps 3-4: actor maximises mean-ensemble Q minus the entropy penalty,
        then alpha is tuned so the policy entropy tracks `target_entropy`."""
        new_action, logp = self.actor(obs)
        q_pi = self.critic(obs, new_action).mean(dim=0)  # (N, 1)
        actor_loss = (alpha.detach() * logp - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        return {
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "entropy": float(-logp.mean().item()),
        }

    def _soft_update_target(self) -> None:
        """Step 5: target <- (1 - tau) * target + tau * critic (EMA)."""
        with torch.no_grad():
            torch._foreach_mul_(self._target_params, 1.0 - self.tau)
            torch._foreach_add_(
                self._target_params, self._critic_params, alpha=self.tau
            )
