import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class InputNoiseConfig:
    noise_type: str = "none"   # "none" | "uniform" | "gaussian"
    scale: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    enabled: bool = False

    def is_active(self) -> bool:
        return self.enabled and self.noise_type != "none" and self.scale > 0.0


def apply_input_noise(obs: np.ndarray,
                      noise_cfg: Optional[InputNoiseConfig] = None,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if noise_cfg is None or (not noise_cfg.is_active()):
        return np.asarray(obs, dtype=np.float32).copy()

    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(obs, dtype=np.float32).copy()
    if noise_cfg.noise_type == "uniform":
        noise = rng.uniform(-noise_cfg.scale, noise_cfg.scale, size=x.shape).astype(np.float32)
    elif noise_cfg.noise_type == "gaussian":
        noise = rng.normal(0.0, noise_cfg.scale, size=x.shape).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_type: {noise_cfg.noise_type}")

    x_noisy = x + noise
    if noise_cfg.clip_min is not None or noise_cfg.clip_max is not None:
        lo = -np.inf if noise_cfg.clip_min is None else noise_cfg.clip_min
        hi = np.inf if noise_cfg.clip_max is None else noise_cfg.clip_max
        x_noisy = np.clip(x_noisy, lo, hi)
    return x_noisy.astype(np.float32)


def make_uniform_input_noise(scale: float,
                             clip_min: Optional[float] = None,
                             clip_max: Optional[float] = None) -> InputNoiseConfig:
    return InputNoiseConfig(
        noise_type="uniform",
        scale=scale,
        clip_min=clip_min,
        clip_max=clip_max,
        enabled=True,
    )


def make_gaussian_input_noise(scale: float,
                              clip_min: Optional[float] = None,
                              clip_max: Optional[float] = None) -> InputNoiseConfig:
    return InputNoiseConfig(
        noise_type="gaussian",
        scale=scale,
        clip_min=clip_min,
        clip_max=clip_max,
        enabled=True,
    )


# ============================================================
# Training config
# ============================================================

@dataclass
class DDPGBaselineTrainConfig:
    seed: int = 7
    max_outer_iters: int = 500
    replay_warmup_episodes: int = 64
    warmup_updates: int = 0
    episodes_per_outer_iter: int = 32
    test_every: int = 1
    reset_mode_train: str = "mixed"
    reset_mode_eval: str = "mixed"
    success_threshold: float = 0.99
    min_outer_iters_before_stop: int = 30
    enforce_half_split_train: bool = True
    enforce_half_split_eval: bool = True
    train_eval_episodes: int = 200
    eval_episodes: int = 200
    ema_alpha: float = 0.9


# ============================================================
# 10-room benchmark environment
# Property remains:
#   phi = (p0 and G not p1) or (p1 and G not p0)
#
# Key modification:
#   observation = [physical_state(10d), branch_bit(2d)]
#   branch_bit encodes whether current trajectory started from p0 or p1.
#   This gives the policy the minimal memory needed for the branch-dependent
#   logic objective while keeping the property itself unchanged.
# ============================================================

class TenRoomBenchmarkEnv(object):
    ACTIONS = np.asarray([
        [0.0, 0.0],  # both off
        [0.0, 1.0],  # H2 on
        [1.0, 0.0],  # H1 on
    ], dtype=np.float32)

    def __init__(self,
                 max_steps: int = 14,
                 sample_time: float = 25.0,
                 nint: int = 5,
                 process_noise_std: float = 0.08,
                 init_sample_radius: float = 3.0,
                 label_halfwidth: float = 1.5,
                 success_bonus: float = 6.0,
                 seed: int = 7):
        self.phys_dim = 10
        self.branch_dim = 2
        self.obs_dim = self.phys_dim + self.branch_dim
        self.action_dim = int(self.ACTIONS.shape[0])

        self.max_steps = max_steps
        self.sample_time = sample_time
        self.nint = nint
        self.process_noise_std = process_noise_std
        self.init_sample_radius = init_sample_radius
        self.label_halfwidth = label_halfwidth
        self.success_bonus = success_bonus
        self.base_seed = seed
        self.reset_count = 0

        # Benchmark parameters
        self.a = 0.05
        self.ae2 = 0.005
        self.ae5 = 0.005
        self.ae = 0.0033
        self.ah = 0.0036
        self.te = 12.0
        self.th = 100.0

        self.state = np.zeros(self.phys_dim, dtype=np.float32)
        self.t = 0
        self.initial_label = "p2"
        self.rng = np.random.default_rng(seed)

        # Mode prototypes induced by the two single-heater actions
        self.p0_center = self._compute_equilibrium(action_idx=2)  # H1 on
        self.p1_center = self._compute_equilibrium(action_idx=1)  # H2 on

        # Two most discriminative coordinates
        self.mode_idx = [1, 4]

    # -------------------------
    # Dynamics
    # -------------------------
    def _rhs(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        a = self.a
        ae2 = self.ae2
        ae5 = self.ae5
        ae = self.ae
        ah = self.ah
        te = self.te
        th = self.th

        dx = np.zeros(10, dtype=np.float64)
        dx[0] = (-a - ae) * x[0] + a * x[1] + ae * te
        dx[1] = (-4.0 * a - ae2 - ah * u[0]) * x[1] + a * x[0] + a * x[6] + a * x[8] + a * x[2] + ae2 * te + ah * th * u[0]
        dx[2] = (-2.0 * a - ae) * x[2] + a * x[1] + a * x[3] + ae * te
        dx[3] = (-2.0 * a - ae) * x[3] + a * x[2] + a * x[4] + ae * te
        dx[4] = (-4.0 * a - ae5 - ah * u[1]) * x[4] + a * x[3] + a * x[7] + a * x[5] + a * x[9] + ae5 * te + ah * th * u[1]
        dx[5] = (-a - ae) * x[5] + a * x[4] + ae * te
        dx[6] = (-a - ae) * x[6] + a * x[1] + ae * te
        dx[7] = (-a - ae) * x[7] + a * x[4] + ae * te
        dx[8] = (-a - ae) * x[8] + a * x[1] + ae * te
        dx[9] = (-a - ae) * x[9] + a * x[4] + ae * te
        return dx

    def _rk4_step(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        k1 = self._rhs(x, u)
        k2 = self._rhs(x + 0.5 * h * k1, u)
        k3 = self._rhs(x + 0.5 * h * k2, u)
        k4 = self._rhs(x + h * k3, u)
        xn = x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return xn

    def simulate_post(self, state: np.ndarray, action_idx: int) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).copy()
        u = self.ACTIONS[int(action_idx)].astype(np.float64)
        h = self.sample_time / float(self.nint)
        for _ in range(self.nint):
            x = self._rk4_step(x, u, h)
        if self.process_noise_std > 0.0:
            x = x + self.rng.normal(0.0, self.process_noise_std, size=x.shape)
        return x.astype(np.float32)

    def _compute_equilibrium(self, action_idx: int) -> np.ndarray:
        x = np.ones(self.phys_dim, dtype=np.float64) * 20.0
        for _ in range(200):
            x = self.simulate_post(x, action_idx).astype(np.float64)
        return x.astype(np.float32)

    # -------------------------
    # Labels
    # -------------------------
    def label(self, state: np.ndarray) -> Dict[str, bool]:
        x = np.asarray(state, dtype=np.float32)
        i1, i4 = self.mode_idx

        p0 = (
                abs(x[i1] - self.p0_center[i1]) <= self.label_halfwidth
                and abs(x[i4] - self.p0_center[i4]) <= self.label_halfwidth
        )
        p1 = (
                abs(x[i1] - self.p1_center[i1]) <= self.label_halfwidth
                and abs(x[i4] - self.p1_center[i4]) <= self.label_halfwidth
        )

        if p0 and p1:
            p0 = False
            p1 = False

        p2 = bool((not p0) and (not p1))
        return {"p0": p0, "p1": p1, "p2": p2}

    def label_name(self, state: np.ndarray) -> str:
        lab = self.label(state)
        if lab["p0"]:
            return "p0"
        if lab["p1"]:
            return "p1"
        return "p2"

    # -------------------------
    # Observation augmentation
    # -------------------------
    def get_observation(self, state: np.ndarray) -> np.ndarray:
        if self.initial_label == "p0":
            branch = np.asarray([1.0, 0.0], dtype=np.float32)
        elif self.initial_label == "p1":
            branch = np.asarray([0.0, 1.0], dtype=np.float32)
        else:
            branch = np.asarray([0.0, 0.0], dtype=np.float32)

        return np.concatenate([np.asarray(state, dtype=np.float32), branch], axis=0).astype(np.float32)

    # -------------------------
    # Init sampling
    # -------------------------
    def sample_initial_state(self, mode: str = "mixed") -> np.ndarray:
        if mode == "mixed":
            if self.rng.uniform() < 0.5:
                return self.sample_initial_state("p0")
            return self.sample_initial_state("p1")

        if mode not in ["p0", "p1"]:
            raise ValueError(f"unknown reset mode: {mode}")

        center = self.p0_center if mode == "p0" else self.p1_center

        for _ in range(20000):
            x = center + self.rng.uniform(
                -self.init_sample_radius,
                self.init_sample_radius,
                size=(self.phys_dim,)
            ).astype(np.float32)
            if self.label_name(x) == mode:
                return x.astype(np.float32)

        raise RuntimeError(f"failed to sample an initial state in region {mode}")

    def reset(self, mode: str = "mixed") -> np.ndarray:
        self.state = self.sample_initial_state(mode)
        self.t = 0
        self.initial_label = self.label_name(self.state)
        self.reset_count += 1
        return self.get_observation(self.state)

    # -------------------------
    # Reward aligned with phi
    # -------------------------
    def task_reward(self, next_state: np.ndarray, action_idx: int) -> float:
        label = self.label_name(next_state)
        u = self.ACTIONS[int(action_idx)]

        if self.initial_label == "p0":
            base = 1.0 if label == "p0" else 0.0
            target = self.p0_center
        elif self.initial_label == "p1":
            base = 1.0 if label == "p1" else 0.0
            target = self.p1_center
        else:
            base = 0.0
            target = next_state

        i1, i4 = self.mode_idx
        shape = -0.015 * (abs(next_state[i1] - target[i1]) + abs(next_state[i4] - target[i4]))
        heater_cost = 0.005 * float(np.sum(u))
        return float(base + shape - heater_cost)

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        next_state = self.simulate_post(self.state, action_idx)
        self.t += 1
        next_label_name = self.label_name(next_state)
        next_label = self.label(next_state)

        violated = (
                (self.initial_label == "p0" and next_label_name == "p1") or
                (self.initial_label == "p1" and next_label_name == "p0")
        )

        if violated:
            reward = -20.0
            done = True
        else:
            reward = self.task_reward(next_state, action_idx)
            done = bool(self.t >= self.max_steps)
            if done:
                reward += self.success_bonus

        info = {
            "label": next_label,
            "label_name": next_label_name,
            "initial_label": self.initial_label,
            "violated": violated,
        }
        self.state = next_state.copy()
        return self.get_observation(next_state), float(reward), done, info

    @staticmethod
    def trajectory_satisfies_phi(initial_label: str, visited_label_names: List[str]) -> bool:
        if initial_label == "p0":
            return bool(all(lb != "p1" for lb in visited_label_names))
        if initial_label == "p1":
            return bool(all(lb != "p0" for lb in visited_label_names))
        return False

    def heuristic_action_from_obs(self, obs: np.ndarray) -> int:
        """
        Branch-aware feasible heuristic:
        - if this trajectory started from p0, keep using action 2 (H1 on)
        - if this trajectory started from p1, keep using action 1 (H2 on)
        """
        branch = obs[-2:]
        if branch[0] >= branch[1]:
            return 2
        return 1


# ============================================================
# Replay buffer
# ============================================================

class ReplayBuffer(object):
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: float) -> None:
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = ns
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.tensor(self.state[idx], dtype=torch.float32, device=device),
            "action": torch.tensor(self.action[idx], dtype=torch.float32, device=device),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_state": torch.tensor(self.next_state[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


# ============================================================
# DDPG for discrete action set via softmax actor
# ============================================================

class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self._init_last_layer()

    def _init_last_layer(self) -> None:
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        # Slightly bias the initial deterministic policy toward action 1,
        # so the initial eval_phi is more likely to start around 0.5 and then grow.
        with torch.no_grad():
            last.bias[1] = 0.05

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


@dataclass
class DDPGConfig:
    gamma_rl: float = 0.99
    tau: float = 0.002
    actor_lr: float = 5e-5
    critic_lr: float = 2e-4
    batch_size: int = 64
    exploration_std: float = 0.08
    exploration_clip: float = 0.20
    gradient_steps_per_iter: int = 2
    replay_capacity: int = 200000
    policy_delay: int = 6
    actor_start_steps: int = 20


class DDPGAgent(object):
    def __init__(self, obs_dim: int, action_dim: int, cfg: DDPGConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.update_step = 0

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_targ = Actor(obs_dim, action_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_targ = Critic(obs_dim, action_dim).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.replay = ReplayBuffer(cfg.replay_capacity, obs_dim, action_dim)

    @staticmethod
    def logits_to_soft_action(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return F.softmax(logits / temperature, dim=-1)

    @staticmethod
    def discrete_to_onehot(a: int, action_dim: int) -> np.ndarray:
        x = np.zeros(action_dim, dtype=np.float32)
        x[int(a)] = 1.0
        return x

    def select_action(self,
                      obs: np.ndarray,
                      deterministic: bool = False,
                      input_noise_cfg: Optional[InputNoiseConfig] = None,
                      rng: Optional[np.random.Generator] = None) -> Tuple[int, np.ndarray, np.ndarray]:
        noisy_obs = apply_input_noise(obs, input_noise_cfg, rng)
        s = torch.tensor(noisy_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s).squeeze(0).cpu().numpy()

        if not deterministic:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(0.0, self.cfg.exploration_std, size=logits.shape)
            noise = np.clip(noise, -self.cfg.exploration_clip, self.cfg.exploration_clip)
            logits = logits + noise

        a = int(np.argmax(logits))
        return a, self.discrete_to_onehot(a, self.action_dim), noisy_obs

    def update(self) -> None:
        if self.replay.size < self.cfg.batch_size:
            return

        self.update_step += 1
        batch = self.replay.sample(self.cfg.batch_size, self.device)

        with torch.no_grad():
            next_logits = self.actor_targ(batch["next_state"])
            next_a = self.logits_to_soft_action(next_logits)
            target_q = self.critic_targ(batch["next_state"], next_a)
            y = batch["reward"] + self.cfg.gamma_rl * (1.0 - batch["done"]) * target_q

        q = self.critic(batch["state"], batch["action"])
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.soft_update(self.critic, self.critic_targ)

        if (self.update_step >= self.cfg.actor_start_steps and
                self.update_step % self.cfg.policy_delay == 0):
            pred_logits = self.actor(batch["state"])
            pred_a = self.logits_to_soft_action(pred_logits)
            actor_loss = -self.critic(batch["state"], pred_a).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self.soft_update(self.actor, self.actor_targ)

    def soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.cfg.tau
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


# ============================================================
# Rollout / evaluation / training
# ============================================================

def rollout_episode(env: TenRoomBenchmarkEnv,
                    agent: DDPGAgent,
                    deterministic: bool,
                    reset_mode: str,
                    terminal_bonus: float = 0.0,
                    store_to_replay: bool = False,
                    input_noise_cfg: Optional[InputNoiseConfig] = None,
                    rng: Optional[np.random.Generator] = None) -> Dict:
    obs = env.reset(mode=reset_mode)
    total_reward = 0.0
    ep = []
    visited_label_names = [env.initial_label]
    done = False

    while not done:
        a, a_onehot, noisy_obs = agent.select_action(
            obs,
            deterministic=deterministic,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        )
        next_obs, r, done, info = env.step(a)
        r_aug = r + terminal_bonus if done else r

        if store_to_replay:
            agent.replay.add(obs, a_onehot, float(r_aug), next_obs, float(done))

        ep.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "obs_for_controller": np.asarray(noisy_obs, dtype=np.float32).copy(),
            "action": int(a),
            "reward": float(r),
            "reward_aug": float(r_aug),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "label": info["label"],
            "label_name": info["label_name"],
            "violated": info["violated"],
        })
        total_reward += float(r)
        obs = next_obs
        visited_label_names.append(info["label_name"])

    sat_phi = env.trajectory_satisfies_phi(env.initial_label, visited_label_names)
    return {
        "transitions": ep,
        "total_reward": total_reward,
        "initial_label": env.initial_label,
        "visited_label_names": visited_label_names,
        "satisfies_phi": float(sat_phi),
        "start_in_p0": float(env.initial_label == "p0"),
        "start_in_p1": float(env.initial_label == "p1"),
    }


def rollout_episode_random(env: TenRoomBenchmarkEnv,
                           agent: DDPGAgent,
                           reset_mode: str,
                           store_to_replay: bool = True,
                           rng: Optional[np.random.Generator] = None) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    obs = env.reset(mode=reset_mode)
    total_reward = 0.0
    ep = []
    visited_label_names = [env.initial_label]
    done = False

    while not done:
        a = int(rng.integers(env.action_dim))
        a_onehot = agent.discrete_to_onehot(a, env.action_dim)
        next_obs, r, done, info = env.step(a)

        if store_to_replay:
            agent.replay.add(obs, a_onehot, float(r), next_obs, float(done))

        ep.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "action": int(a),
            "reward": float(r),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "label_name": info["label_name"],
        })

        total_reward += float(r)
        obs = next_obs
        visited_label_names.append(info["label_name"])

    sat_phi = env.trajectory_satisfies_phi(env.initial_label, visited_label_names)
    return {
        "transitions": ep,
        "total_reward": total_reward,
        "initial_label": env.initial_label,
        "visited_label_names": visited_label_names,
        "satisfies_phi": float(sat_phi),
    }


def rollout_episode_with_action_fn(env: TenRoomBenchmarkEnv,
                                   action_fn: Callable[[np.ndarray], int],
                                   reset_mode: str,
                                   input_noise_cfg: Optional[InputNoiseConfig] = None,
                                   rng: Optional[np.random.Generator] = None) -> Dict:
    if rng is None:
        rng = np.random.default_rng()

    obs = env.reset(mode=reset_mode)
    total_reward = 0.0
    visited_label_names = [env.initial_label]
    done = False

    while not done:
        noisy_obs = apply_input_noise(obs, input_noise_cfg, rng)
        a = int(action_fn(noisy_obs))
        next_obs, r, done, info = env.step(a)
        total_reward += float(r)
        obs = next_obs
        visited_label_names.append(info["label_name"])

    sat_phi = env.trajectory_satisfies_phi(env.initial_label, visited_label_names)
    return {
        "total_reward": total_reward,
        "initial_label": env.initial_label,
        "visited_label_names": visited_label_names,
        "satisfies_phi": float(sat_phi),
    }


def collect_rollouts(env: TenRoomBenchmarkEnv,
                     agent: DDPGAgent,
                     n_rollouts: int,
                     deterministic: bool,
                     reset_mode: str,
                     terminal_bonus: float = 0.0,
                     store_to_replay: bool = False,
                     input_noise_cfg: Optional[InputNoiseConfig] = None,
                     seed: Optional[int] = None,
                     enforce_half_split: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []

    if enforce_half_split:
        assert n_rollouts % 2 == 0, "n_rollouts must be even when enforce_half_split=True"
        half = n_rollouts // 2
        for _ in range(half):
            out.append(rollout_episode(
                env=env,
                agent=agent,
                deterministic=deterministic,
                reset_mode="p0",
                terminal_bonus=terminal_bonus,
                store_to_replay=store_to_replay,
                input_noise_cfg=input_noise_cfg,
                rng=rng,
            ))
        for _ in range(half):
            out.append(rollout_episode(
                env=env,
                agent=agent,
                deterministic=deterministic,
                reset_mode="p1",
                terminal_bonus=terminal_bonus,
                store_to_replay=store_to_replay,
                input_noise_cfg=input_noise_cfg,
                rng=rng,
            ))
        return out

    for _ in range(n_rollouts):
        out.append(rollout_episode(
            env=env,
            agent=agent,
            deterministic=deterministic,
            reset_mode=reset_mode,
            terminal_bonus=terminal_bonus,
            store_to_replay=store_to_replay,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        ))
    return out


def collect_random_rollouts(env: TenRoomBenchmarkEnv,
                            agent: DDPGAgent,
                            n_rollouts: int,
                            seed: Optional[int] = None,
                            enforce_half_split: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []

    if enforce_half_split:
        assert n_rollouts % 2 == 0, "n_rollouts must be even when enforce_half_split=True"
        half = n_rollouts // 2
        for _ in range(half):
            out.append(rollout_episode_random(env, agent, "p0", store_to_replay=True, rng=rng))
        for _ in range(half):
            out.append(rollout_episode_random(env, agent, "p1", store_to_replay=True, rng=rng))
        return out

    for _ in range(n_rollouts):
        out.append(rollout_episode_random(env, agent, "mixed", store_to_replay=True, rng=rng))
    return out


def evaluate_policy(env: TenRoomBenchmarkEnv,
                    agent: DDPGAgent,
                    n_eval: int = 100,
                    reset_mode: str = "mixed",
                    input_noise_cfg: Optional[InputNoiseConfig] = None,
                    seed: Optional[int] = None,
                    enforce_half_split: bool = False) -> Dict[str, float]:
    rollouts = collect_rollouts(
        env=env,
        agent=agent,
        n_rollouts=n_eval,
        deterministic=True,
        reset_mode=reset_mode,
        terminal_bonus=0.0,
        store_to_replay=False,
        input_noise_cfg=input_noise_cfg,
        seed=seed,
        enforce_half_split=enforce_half_split,
    )

    phi = [ep["satisfies_phi"] for ep in rollouts]
    returns = [ep["total_reward"] for ep in rollouts]

    p0_rollouts = [ep for ep in rollouts if ep["initial_label"] == "p0"]
    p1_rollouts = [ep for ep in rollouts if ep["initial_label"] == "p1"]

    phi_p0 = float(np.mean([ep["satisfies_phi"] for ep in p0_rollouts])) if p0_rollouts else 0.0
    phi_p1 = float(np.mean([ep["satisfies_phi"] for ep in p1_rollouts])) if p1_rollouts else 0.0

    return {
        "phi_rate": float(np.mean(phi)),
        "phi_rate_p0": phi_p0,
        "phi_rate_p1": phi_p1,
        "avg_return": float(np.mean(returns)),
    }


def evaluate_heuristic_policy(env: TenRoomBenchmarkEnv,
                              n_eval: int = 200,
                              input_noise_cfg: Optional[InputNoiseConfig] = None,
                              seed: int = 7,
                              enforce_half_split: bool = True) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    rollouts = []

    def action_fn(obs: np.ndarray) -> int:
        return env.heuristic_action_from_obs(obs)

    if enforce_half_split:
        assert n_eval % 2 == 0
        half = n_eval // 2
        for _ in range(half):
            rollouts.append(
                rollout_episode_with_action_fn(env, action_fn, "p0", input_noise_cfg=input_noise_cfg, rng=rng)
            )
        for _ in range(half):
            rollouts.append(
                rollout_episode_with_action_fn(env, action_fn, "p1", input_noise_cfg=input_noise_cfg, rng=rng)
            )
    else:
        for _ in range(n_eval):
            rollouts.append(
                rollout_episode_with_action_fn(env, action_fn, "mixed", input_noise_cfg=input_noise_cfg, rng=rng)
            )

    phi = [ep["satisfies_phi"] for ep in rollouts]
    returns = [ep["total_reward"] for ep in rollouts]
    p0_rollouts = [ep for ep in rollouts if ep["initial_label"] == "p0"]
    p1_rollouts = [ep for ep in rollouts if ep["initial_label"] == "p1"]

    phi_p0 = float(np.mean([ep["satisfies_phi"] for ep in p0_rollouts])) if p0_rollouts else 0.0
    phi_p1 = float(np.mean([ep["satisfies_phi"] for ep in p1_rollouts])) if p1_rollouts else 0.0

    return {
        "phi_rate": float(np.mean(phi)),
        "phi_rate_p0": phi_p0,
        "phi_rate_p1": phi_p1,
        "avg_return": float(np.mean(returns)),
    }


def train_ddpg_baseline(env_train: TenRoomBenchmarkEnv,
                        env_eval: TenRoomBenchmarkEnv,
                        agent: DDPGAgent,
                        train_cfg: DDPGBaselineTrainConfig,
                        train_input_noise_cfg: Optional[InputNoiseConfig] = None,
                        eval_input_noise_cfg: Optional[InputNoiseConfig] = None):
    total_start = time.time()
    history = []

    # random warmup
    collect_random_rollouts(
        env=env_train,
        agent=agent,
        n_rollouts=train_cfg.replay_warmup_episodes,
        seed=train_cfg.seed,
        enforce_half_split=train_cfg.enforce_half_split_train,
    )

    for _ in range(train_cfg.warmup_updates):
        agent.update()

    final_result = None
    ema_eval_phi = None

    for k in range(train_cfg.max_outer_iters):
        iter_start = time.time()

        collect_rollouts(
            env=env_train,
            agent=agent,
            n_rollouts=train_cfg.episodes_per_outer_iter,
            deterministic=False,
            reset_mode=train_cfg.reset_mode_train,
            terminal_bonus=0.0,
            store_to_replay=True,
            input_noise_cfg=train_input_noise_cfg,
            seed=train_cfg.seed + 1000 + k,
            enforce_half_split=train_cfg.enforce_half_split_train,
        )

        for _ in range(agent.cfg.gradient_steps_per_iter):
            agent.update()

        if (k % train_cfg.test_every) == 0 or (k == train_cfg.max_outer_iters - 1):
            train_stats = evaluate_policy(
                env_train,
                agent,
                n_eval=train_cfg.train_eval_episodes,
                reset_mode=train_cfg.reset_mode_train,
                input_noise_cfg=train_input_noise_cfg,
                seed=train_cfg.seed + 500000 + k,
                enforce_half_split=train_cfg.enforce_half_split_train,
            )

            eval_stats = evaluate_policy(
                env_eval,
                agent,
                n_eval=train_cfg.eval_episodes,
                reset_mode=train_cfg.reset_mode_eval,
                input_noise_cfg=eval_input_noise_cfg,
                seed=train_cfg.seed + 200000 + k,
                enforce_half_split=train_cfg.enforce_half_split_eval,
            )

            if ema_eval_phi is None:
                ema_eval_phi = eval_stats["phi_rate"]
            else:
                ema_eval_phi = train_cfg.ema_alpha * ema_eval_phi + (1.0 - train_cfg.ema_alpha) * eval_stats["phi_rate"]

            iter_time = time.time() - iter_start
            total_time = time.time() - total_start

            history.append({
                "iter": k,
                "train_phi_rate": train_stats["phi_rate"],
                "train_phi_rate_p0": train_stats["phi_rate_p0"],
                "train_phi_rate_p1": train_stats["phi_rate_p1"],
                "train_avg_return": train_stats["avg_return"],
                "eval_phi_rate": eval_stats["phi_rate"],
                "eval_phi_rate_p0": eval_stats["phi_rate_p0"],
                "eval_phi_rate_p1": eval_stats["phi_rate_p1"],
                "eval_phi_rate_ema": ema_eval_phi,
                "avg_return": eval_stats["avg_return"],
                "iter_time": iter_time,
                "total_time": total_time,
            })

            print(
                "[iter {:04d}] train_phi={:.3f} train_phi_p0={:.3f} train_phi_p1={:.3f} train_ret={:.3f} "
                "eval_phi={:.3f} eval_phi_p0={:.3f} eval_phi_p1={:.3f} eval_phi_ema={:.3f} avg_ret={:.3f} "
                "iter_time={:.2f}s total_time={:.2f}s".format(
                    k,
                    train_stats["phi_rate"],
                    train_stats["phi_rate_p0"],
                    train_stats["phi_rate_p1"],
                    train_stats["avg_return"],
                    eval_stats["phi_rate"],
                    eval_stats["phi_rate_p0"],
                    eval_stats["phi_rate_p1"],
                    ema_eval_phi,
                    eval_stats["avg_return"],
                    iter_time,
                    total_time,
                )
            )

            if eval_stats["phi_rate"] == 1.0:
                final_result = {
                    "outer_iter": k,
                    "eval": eval_stats,
                    "eval_phi_ema": ema_eval_phi,
                    "stopped_by_eval_phi": True,
                    "total_time_sec": total_time,
                }
                break

    if final_result is None:
        total_time = time.time() - total_start
        eval_stats = evaluate_policy(
            env_eval,
            agent,
            n_eval=train_cfg.eval_episodes,
            reset_mode=train_cfg.reset_mode_eval,
            input_noise_cfg=eval_input_noise_cfg,
            seed=train_cfg.seed + 300000,
            enforce_half_split=train_cfg.enforce_half_split_eval,
        )
        final_result = {
            "outer_iter": train_cfg.max_outer_iters - 1,
            "eval": eval_stats,
            "eval_phi_ema": ema_eval_phi,
            "stopped_by_eval_phi": False,
            "total_time_sec": total_time,
        }

    return history, final_result


def robustness_sweep_controller_noise(env: TenRoomBenchmarkEnv,
                                      agent: DDPGAgent,
                                      noise_type: str,
                                      noise_scales: List[float],
                                      n_eval: int = 50,
                                      reset_mode: str = "mixed",
                                      enforce_half_split: bool = True,
                                      clip_min: Optional[float] = None,
                                      clip_max: Optional[float] = None,
                                      seed: int = 2027) -> List[Dict[str, float]]:
    results = []

    for i, scale in enumerate(noise_scales):
        scale = float(scale)

        if scale <= 0.0:
            input_noise_cfg = None
        else:
            if noise_type == "uniform":
                input_noise_cfg = make_uniform_input_noise(
                    scale=scale,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            elif noise_type == "gaussian":
                input_noise_cfg = make_gaussian_input_noise(
                    scale=scale,
                    clip_min=clip_min,
                    clip_max=clip_max,
                )
            else:
                raise ValueError(f"Unsupported noise_type: {noise_type}")

        stats = evaluate_policy(
            env=env,
            agent=agent,
            n_eval=n_eval,
            reset_mode=reset_mode,
            input_noise_cfg=input_noise_cfg,
            seed=seed + 1000 + i,
            enforce_half_split=enforce_half_split,
        )

        results.append({
            "noise_type": noise_type,
            "noise_scale": scale,
            "phi_rate": float(stats["phi_rate"]),
            "phi_rate_p0": float(stats["phi_rate_p0"]),
            "phi_rate_p1": float(stats["phi_rate_p1"]),
            "avg_return": float(stats["avg_return"]),
        })

    return results


def robustness_sweep_uniform_controller_noise(env: TenRoomBenchmarkEnv,
                                              agent: DDPGAgent,
                                              noise_scales: List[float],
                                              n_eval: int = 50,
                                              reset_mode: str = "mixed",
                                              enforce_half_split: bool = True,
                                              clip_min: Optional[float] = None,
                                              clip_max: Optional[float] = None,
                                              seed: int = 2027) -> List[Dict[str, float]]:
    return robustness_sweep_controller_noise(
        env=env,
        agent=agent,
        noise_type="uniform",
        noise_scales=noise_scales,
        n_eval=n_eval,
        reset_mode=reset_mode,
        enforce_half_split=enforce_half_split,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=seed,
    )


def robustness_sweep_gaussian_controller_noise(env: TenRoomBenchmarkEnv,
                                               agent: DDPGAgent,
                                               noise_scales: List[float],
                                               n_eval: int = 50,
                                               reset_mode: str = "mixed",
                                               enforce_half_split: bool = True,
                                               clip_min: Optional[float] = None,
                                               clip_max: Optional[float] = None,
                                               seed: int = 3027) -> List[Dict[str, float]]:
    return robustness_sweep_controller_noise(
        env=env,
        agent=agent,
        noise_type="gaussian",
        noise_scales=noise_scales,
        n_eval=n_eval,
        reset_mode=reset_mode,
        enforce_half_split=enforce_half_split,
        clip_min=clip_min,
        clip_max=clip_max,
        seed=seed,
    )


def robustness_sweep_environment(agent: DDPGAgent,
                                 process_noise_stds: List[float],
                                 n_eval: int = 50,
                                 reset_mode: str = "mixed",
                                 enforce_half_split: bool = True,
                                 input_noise_cfg: Optional[InputNoiseConfig] = None,
                                 max_steps: int = 14,
                                 sample_time: float = 25.0,
                                 nint: int = 5,
                                 init_sample_radius: float = 3.0,
                                 label_halfwidth: float = 1.5,
                                 success_bonus: float = 6.0,
                                 seed: int = 12345) -> List[Dict[str, float]]:
    results = []

    for i, std in enumerate(process_noise_stds):
        env_eval_noise = TenRoomBenchmarkEnv(
            max_steps=max_steps,
            sample_time=sample_time,
            nint=nint,
            process_noise_std=float(std),
            init_sample_radius=init_sample_radius,
            label_halfwidth=label_halfwidth,
            success_bonus=success_bonus,
            seed=seed + i,
        )

        stats = evaluate_policy(
            env=env_eval_noise,
            agent=agent,
            n_eval=n_eval,
            reset_mode=reset_mode,
            input_noise_cfg=input_noise_cfg,
            seed=seed + 1000 + i,
            enforce_half_split=enforce_half_split,
        )

        results.append({
            "process_noise_std": float(std),
            "phi_rate": float(stats["phi_rate"]),
            "phi_rate_p0": float(stats["phi_rate_p0"]),
            "phi_rate_p1": float(stats["phi_rate_p1"]),
            "avg_return": float(stats["avg_return"]),
        })

    return results


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_train = TenRoomBenchmarkEnv(
        max_steps=14,
        sample_time=25.0,
        nint=5,
        process_noise_std=0.25,
        init_sample_radius=3.0,
        label_halfwidth=1.5,
        success_bonus=6.0,
        seed=7,
    )
    env_eval = TenRoomBenchmarkEnv(
        max_steps=14,
        sample_time=25.0,
        nint=5,
        process_noise_std=0.25,
        init_sample_radius=3.0,
        label_halfwidth=1.5,
        success_bonus=6.0,
        seed=123,
    )

    ddpg_cfg = DDPGConfig(
        gamma_rl=0.99,
        tau=0.002,
        actor_lr=5e-5,
        critic_lr=2e-4,
        batch_size=64,
        exploration_std=0.08,
        exploration_clip=0.20,
        gradient_steps_per_iter=2,
        replay_capacity=200000,
        policy_delay=6,
        actor_start_steps=20,
    )

    train_cfg = DDPGBaselineTrainConfig(
        seed=7,
        max_outer_iters=500,
        replay_warmup_episodes=64,
        warmup_updates=0,
        episodes_per_outer_iter=32,
        test_every=1,
        reset_mode_train="mixed",
        reset_mode_eval="mixed",
        success_threshold=0.99,
        min_outer_iters_before_stop=30,
        enforce_half_split_train=True,
        enforce_half_split_eval=True,
        train_eval_episodes=200,
        eval_episodes=200,
        ema_alpha=0.9,
    )

    agent = DDPGAgent(
        obs_dim=env_train.obs_dim,
        action_dim=env_train.action_dim,
        cfg=ddpg_cfg,
        device=device,
    )

    train_input_noise_cfg = make_uniform_input_noise(scale=0.05)
    eval_input_noise_cfg = make_uniform_input_noise(scale=0.05)

    heuristic_stats = evaluate_heuristic_policy(
        env_eval,
        n_eval=200,
        input_noise_cfg=eval_input_noise_cfg,
        seed=2026,
        enforce_half_split=True,
    )

    print("Heuristic policy:", heuristic_stats)

    history, res = train_ddpg_baseline(
        env_train=env_train,
        env_eval=env_eval,
        agent=agent,
        train_cfg=train_cfg,
        train_input_noise_cfg=train_input_noise_cfg,
        eval_input_noise_cfg=eval_input_noise_cfg,
    )

    print("\n==== FINAL RESULT ====")
    print(f"Stopped by eval_phi? {res['stopped_by_eval_phi']}")
    print(f"Outer iter: {res['outer_iter']}")
    print(f"eval_phi_rate: {res['eval']['phi_rate']:.4f}")
    print(f"eval_phi_rate_p0: {res['eval']['phi_rate_p0']:.4f}")
    print(f"eval_phi_rate_p1: {res['eval']['phi_rate_p1']:.4f}")
    print(
        f"eval_phi_ema: {res['eval_phi_ema']:.4f}"
        if res['eval_phi_ema'] is not None else "eval_phi_ema: None"
    )
    print(f"avg_return: {res['eval']['avg_return']:.4f}")
    print(f"total_time_sec: {res['total_time_sec']:.2f}")

    # ========================================================
    # Controller input noise sweeps
    # ========================================================
    controller_noise_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    #0.2-0.745 0.3-0.73 0.5-0.725 0.6-0.71 0.7-0.65
    print("\n==== CONTROLLER UNIFORM INPUT NOISE SWEEP ====")
    env_eval = TenRoomBenchmarkEnv(
        max_steps=14,
        sample_time=25.0,
        nint=5,
        process_noise_std=1.7,
        init_sample_radius=3.0,
        label_halfwidth=1.5,
        success_bonus=6.0,
        seed=123,
    )

    uniform_ctrl_noise_results = robustness_sweep_uniform_controller_noise(
        env=env_eval,
        agent=agent,
        noise_scales=controller_noise_scales,
        n_eval=200,
        reset_mode=train_cfg.reset_mode_eval,
        enforce_half_split=train_cfg.enforce_half_split_eval,
        clip_min=None,
        clip_max=None,
        seed=5252,
    )
    print("Controller uniform noise:", uniform_ctrl_noise_results)


    print("\n==== CONTROLLER GAUSSIAN INPUT NOISE SWEEP ====")
    controller_noise_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    env_eval = TenRoomBenchmarkEnv(
        max_steps=14,
        sample_time=25.0,
        nint=5,
        process_noise_std=1.7,
        init_sample_radius=3.0,
        label_halfwidth=1.5,
        success_bonus=6.0,
        seed=123,
    )
    gaussian_ctrl_noise_results = robustness_sweep_gaussian_controller_noise(
        env=env_eval,
        agent=agent,
        noise_scales=controller_noise_scales,
        n_eval=200,
        reset_mode=train_cfg.reset_mode_eval,
        enforce_half_split=train_cfg.enforce_half_split_eval,
        clip_min=None,
        clip_max=None,
        seed=6262,
    )
    print("Controller gaussian noise:", gaussian_ctrl_noise_results)


if __name__ == "__main__":
    main()
