import math
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
# Training configs
# ============================================================

@dataclass
class BaselineTrainConfig:
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


@dataclass
class PACGuidedTrainConfig:
    seed: int = 7
    max_outer_iters: int = 500

    # replay / RL
    replay_warmup_episodes: int = 64
    warmup_updates: int = 0
    episodes_per_outer_iter: int = 32

    # evaluation
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

    # PAC-guided
    N_cert: int = 64
    beta: float = 0.05
    p_min: float = 0.90
    gamma_bc: float = 0.01
    horizon_T: int = 14
    lambda_bc1: float = 25.0
    lambda_bc2: float = 25.0


@dataclass
class CertificateNNConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 500
    lambda_nonneg: float = 5.0
    lambda_pre: float = 5.0
    lambda_post: float = 5.0
    lambda_dyn: float = 5.0
    lambda_c: float = 3.0
    tol: float = 1e-3
    tol_c: float = 1e-3


# ============================================================
# 10-room benchmark environment
# Property:
#   phi = (p0 and G not p1) or (p1 and G not p0)
#
# observation = [physical_state(10d), branch_bit(2d)]
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
                 process_noise_std: float = 0.25,
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
        branch = obs[-2:]
        if branch[0] >= branch[1]:
            return 2
        return 1



# ============================================================
# TD3-style actor-critic for discrete 10-room actions
# ============================================================

class ReplayBuffer(object):
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, ns, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = ns
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.tensor(self.state[idx], dtype=torch.float32, device=device),
            "action": torch.tensor(self.action[idx], dtype=torch.float32, device=device),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_state": torch.tensor(self.next_state[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


@dataclass
class TD3Config:
    gamma_rl: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    exploration_std: float = 0.10
    exploration_clip: float = 0.25
    target_policy_noise: float = 0.10
    target_noise_clip: float = 0.25
    policy_delay: int = 2
    gradient_steps_per_iter: int = 50
    replay_capacity: int = 200000


class TD3Agent(object):
    supports_random_warmup = True

    def __init__(self, state_dim, action_dim, cfg, device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.total_it = 0

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_targ = Actor(state_dim, action_dim).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_targ = Critic(state_dim, action_dim).to(device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_targ = Critic(state_dim, action_dim).to(device)
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )
        self.replay = ReplayBuffer(cfg.replay_capacity, state_dim, action_dim)

    @staticmethod
    def logits_to_soft_action(logits, temperature=1.0):
        return F.softmax(logits / temperature, dim=-1)

    @staticmethod
    def discrete_to_onehot(a, action_dim):
        x = np.zeros(action_dim, dtype=np.float32)
        x[int(a)] = 1.0
        return x

    def select_action(self, state, deterministic=False, input_noise_cfg=None, rng=None):
        noisy_state = apply_input_noise(state, input_noise_cfg, rng)
        s = torch.tensor(noisy_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s).squeeze(0).cpu().numpy()

        if not deterministic:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(0.0, self.cfg.exploration_std, size=logits.shape)
            noise = np.clip(noise, -self.cfg.exploration_clip, self.cfg.exploration_clip)
            logits = logits + noise

        a = int(np.argmax(logits))
        return a, self.discrete_to_onehot(a, self.action_dim), noisy_state

    def store_transition(self, s, action_info, r, ns, d):
        self.replay.add(s, np.asarray(action_info, dtype=np.float32), r, ns, d)

    def store_random_transition(self, s, a, r, ns, d):
        self.replay.add(s, self.discrete_to_onehot(a, self.action_dim), r, ns, d)

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return

        self.total_it += 1
        batch = self.replay.sample(self.cfg.batch_size, self.device)
        with torch.no_grad():
            next_logits = self.actor_targ(batch["next_state"])
            noise = torch.randn_like(next_logits) * self.cfg.target_policy_noise
            noise = torch.clamp(noise, -self.cfg.target_noise_clip, self.cfg.target_noise_clip)
            next_logits = next_logits + noise
            next_a = self.logits_to_soft_action(next_logits)
            target_q1 = self.critic1_targ(batch["next_state"], next_a)
            target_q2 = self.critic2_targ(batch["next_state"], next_a)
            target_q = torch.min(target_q1, target_q2)
            y = batch["reward"] + self.cfg.gamma_rl * (1.0 - batch["done"]) * target_q

        q1 = self.critic1(batch["state"], batch["action"])
        q2 = self.critic2(batch["state"], batch["action"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        if (self.total_it % self.cfg.policy_delay) != 0:
            return

        pred_logits = self.actor(batch["state"])
        pred_a = self.logits_to_soft_action(pred_logits)
        actor_loss = -self.critic1(batch["state"], pred_a).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.actor_targ)
        self.soft_update(self.critic1, self.critic1_targ)
        self.soft_update(self.critic2, self.critic2_targ)

    def soft_update(self, src, tgt):
        tau = self.cfg.tau
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
# ============================================================
# PAC-guided certificate synthesis (on physical state only)
# ============================================================

@dataclass
class CertificateResult:
    feasible: bool
    model_state: Optional[Dict]
    c_bc: float
    gamma_bc: float
    T_bc: int
    u_local: float
    p_lb: float
    n_pre: int
    n_post: int
    n_trans: int
    loss_nonneg: float
    loss_pre: float
    loss_post: float
    loss_dyn: float
    loss_obj: float
    max_violation: float
    epochs_used: int
    status: str


@dataclass
class DualCertificateSummary:
    cert_01: CertificateResult
    cert_10: CertificateResult
    p_lb_phi: float
    w_p0: float
    w_p1: float


class CertificateNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_c = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def c_bc(self) -> torch.Tensor:
        return F.softplus(self.raw_c)


def dedup_rows(X: np.ndarray, decimals: int = 6) -> np.ndarray:
    if X is None or X.shape[0] == 0:
        return X
    Xr = np.round(X.astype(np.float64), decimals=decimals)
    _, idx = np.unique(Xr, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx]


def extract_sampled_sets_for_pair(cert_rollouts: List[Dict], pre_name: str, post_name: str):
    pre_states = []
    post_states = []
    transitions = []
    all_states = []

    for ep in cert_rollouts:
        seq = ep["transitions"]
        if len(seq) == 0:
            continue

        for tr in seq:
            s = tr["state_phys"]
            ns = tr["next_state_phys"]
            lbl = tr["label"]

            all_states.append(s)
            if lbl[pre_name]:
                pre_states.append(s)
            if lbl[post_name]:
                post_states.append(s)
            transitions.append((s, ns))

        final_state = seq[-1]["next_state_phys"]
        final_label = seq[-1]["next_label"]

        all_states.append(final_state)
        if final_label[pre_name]:
            pre_states.append(final_state)
        if final_label[post_name]:
            post_states.append(final_state)

    if len(all_states) == 0:
        return None, None, None, None

    X_all = np.asarray(all_states, dtype=np.float32)
    X_pre = np.asarray(pre_states, dtype=np.float32) if len(pre_states) > 0 else np.zeros((0, X_all.shape[1]), dtype=np.float32)
    X_post = np.asarray(post_states, dtype=np.float32) if len(post_states) > 0 else np.zeros((0, X_all.shape[1]), dtype=np.float32)
    return dedup_rows(X_all), dedup_rows(X_pre), dedup_rows(X_post), transitions


def solve_pair_certificate_nn(cert_rollouts: List[Dict],
                              pre_name: str,
                              post_name: str,
                              horizon_T: int,
                              gamma_bc: float,
                              nn_cfg: CertificateNNConfig,
                              device: torch.device) -> CertificateResult:
    extracted = extract_sampled_sets_for_pair(cert_rollouts, pre_name, post_name)
    if extracted[0] is None:
        return CertificateResult(
            feasible=False,
            model_state=None,
            c_bc=1.0,
            gamma_bc=gamma_bc,
            T_bc=horizon_T,
            u_local=1.0,
            p_lb=0.0,
            n_pre=0,
            n_post=0,
            n_trans=0,
            loss_nonneg=float("inf"),
            loss_pre=float("inf"),
            loss_post=float("inf"),
            loss_dyn=float("inf"),
            loss_obj=1.0,
            max_violation=float("inf"),
            epochs_used=0,
            status="no-data",
        )

    X_all, X_pre, X_post, transitions = extracted
    if len(transitions) == 0:
        return CertificateResult(
            feasible=False,
            model_state=None,
            c_bc=1.0,
            gamma_bc=gamma_bc,
            T_bc=horizon_T,
            u_local=1.0,
            p_lb=0.0,
            n_pre=int(X_pre.shape[0]),
            n_post=int(X_post.shape[0]),
            n_trans=0,
            loss_nonneg=float("inf"),
            loss_pre=float("inf"),
            loss_post=float("inf"),
            loss_dyn=float("inf"),
            loss_obj=1.0,
            max_violation=float("inf"),
            epochs_used=0,
            status="no-transitions",
        )

    if X_post.shape[0] == 0:
        return CertificateResult(
            feasible=True,
            model_state=None,
            c_bc=0.0,
            gamma_bc=gamma_bc,
            T_bc=horizon_T,
            u_local=min(1.0, gamma_bc),
            p_lb=max(0.0, 1.0 - min(1.0, gamma_bc)),
            n_pre=int(X_pre.shape[0]),
            n_post=0,
            n_trans=len(transitions),
            loss_nonneg=0.0,
            loss_pre=0.0,
            loss_post=0.0,
            loss_dyn=0.0,
            loss_obj=0.0,
            max_violation=0.0,
            epochs_used=0,
            status="no-post-direct-c-zero",
        )

    trans_arr = []
    state_dim = int(X_all.shape[1])
    for s, ns in transitions:
        trans_arr.append(np.concatenate([s, ns], axis=0))
    trans_arr = dedup_rows(np.asarray(trans_arr, dtype=np.float32))
    transitions = [(row[:state_dim], row[state_dim:]) for row in trans_arr]

    model = CertificateNet(state_dim=state_dim, hidden_dim=nn_cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=nn_cfg.lr)

    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    X_pre_t = torch.tensor(X_pre, dtype=torch.float32, device=device) if X_pre.shape[0] > 0 else None
    X_post_t = torch.tensor(X_post, dtype=torch.float32, device=device)
    X_s_t = torch.tensor(np.asarray([s for s, _ in transitions], dtype=np.float32), dtype=torch.float32, device=device)
    X_ns_t = torch.tensor(np.asarray([ns for _, ns in transitions], dtype=np.float32), dtype=torch.float32, device=device)

    final_loss_nonneg = float("inf")
    final_loss_pre = float("inf")
    final_loss_post = float("inf")
    final_loss_dyn = float("inf")
    final_loss_obj = float("inf")
    epochs_used = 0

    for epoch in range(nn_cfg.epochs):
        opt.zero_grad()
        eta_all = model(X_all_t)
        eta_s = model(X_s_t)
        eta_ns = model(X_ns_t)
        c_bc_t = model.c_bc()

        loss_nonneg = F.relu(-eta_all).mean()
        if X_pre_t is not None and X_pre_t.shape[0] > 0:
            eta_pre = model(X_pre_t)
            loss_pre = F.relu(eta_pre - gamma_bc).mean()
        else:
            loss_pre = torch.tensor(0.0, device=device)

        eta_post = model(X_post_t)
        loss_post = F.relu(1.0 - eta_post).mean()
        loss_dyn = F.relu(eta_ns - eta_s - c_bc_t).mean()
        loss_obj = c_bc_t

        loss = (
                nn_cfg.lambda_nonneg * loss_nonneg
                + nn_cfg.lambda_pre * loss_pre
                + nn_cfg.lambda_post * loss_post
                + nn_cfg.lambda_dyn * loss_dyn
                + nn_cfg.lambda_c * loss_obj
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            eta_all_chk = model(X_all_t)
            eta_s_chk = model(X_s_t)
            eta_ns_chk = model(X_ns_t)
            c_bc_chk = model.c_bc()

            final_loss_nonneg = float(F.relu(-eta_all_chk).mean().cpu().item())
            if X_pre_t is not None and X_pre_t.shape[0] > 0:
                eta_pre_chk = model(X_pre_t)
                final_loss_pre = float(F.relu(eta_pre_chk - gamma_bc).mean().cpu().item())
            else:
                final_loss_pre = 0.0
            eta_post_chk = model(X_post_t)
            final_loss_post = float(F.relu(1.0 - eta_post_chk).mean().cpu().item())
            final_loss_dyn = float(F.relu(eta_ns_chk - eta_s_chk - c_bc_chk).mean().cpu().item())
            final_loss_obj = float(c_bc_chk.cpu().item())

        epochs_used = epoch + 1
        if (final_loss_nonneg <= nn_cfg.tol and
                final_loss_pre <= nn_cfg.tol and
                final_loss_post <= nn_cfg.tol and
                final_loss_dyn <= nn_cfg.tol and
                final_loss_obj <= nn_cfg.tol_c):
            break

    constraints_satisfied = (
            final_loss_nonneg <= nn_cfg.tol and
            final_loss_pre <= nn_cfg.tol and
            final_loss_post <= nn_cfg.tol and
            final_loss_dyn <= nn_cfg.tol
    )

    with torch.no_grad():
        c_bc = float(model.c_bc().cpu().item())

    if constraints_satisfied:
        u_local = min(1.0, gamma_bc + c_bc * horizon_T)
        p_lb = max(0.0, 1.0 - u_local)
        feasible = True
        status = "constraints-satisfied"
    else:
        u_local = 1.0
        p_lb = 0.0
        feasible = False
        status = "epoch-limit-infeasible"

    max_violation = max(final_loss_nonneg, final_loss_pre, final_loss_post, final_loss_dyn)
    return CertificateResult(
        feasible=feasible,
        model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        c_bc=c_bc,
        gamma_bc=gamma_bc,
        T_bc=horizon_T,
        u_local=u_local,
        p_lb=p_lb,
        n_pre=int(X_pre.shape[0]),
        n_post=int(X_post.shape[0]),
        n_trans=len(transitions),
        loss_nonneg=final_loss_nonneg,
        loss_pre=final_loss_pre,
        loss_post=final_loss_post,
        loss_dyn=final_loss_dyn,
        loss_obj=final_loss_obj,
        max_violation=max_violation,
        epochs_used=epochs_used,
        status=status,
    )


def filter_rollouts_by_initial_label(cert_rollouts: List[Dict], init_label: str) -> List[Dict]:
    return [ep for ep in cert_rollouts if ep["initial_label"] == init_label]


def solve_dual_certificates(cert_rollouts: List[Dict],
                            horizon_T: int,
                            gamma_bc: float,
                            nn_cfg: CertificateNNConfig,
                            device: torch.device) -> DualCertificateSummary:
    rollouts_p0 = filter_rollouts_by_initial_label(cert_rollouts, "p0")
    rollouts_p1 = filter_rollouts_by_initial_label(cert_rollouts, "p1")

    cert_01 = solve_pair_certificate_nn(
        rollouts_p0,
        "p0",
        "p1",
        horizon_T,
        gamma_bc,
        nn_cfg,
        device,
    )

    cert_10 = solve_pair_certificate_nn(
        rollouts_p1,
        "p1",
        "p0",
        horizon_T,
        gamma_bc,
        nn_cfg,
        device,
    )

    n0 = len(rollouts_p0)
    n1 = len(rollouts_p1)
    n = max(1, n0 + n1)

    w_p0 = float(n0) / float(n)
    w_p1 = float(n1) / float(n)

    p_lb_phi = w_p0 * cert_01.p_lb + w_p1 * cert_10.p_lb

    return DualCertificateSummary(
        cert_01=cert_01,
        cert_10=cert_10,
        p_lb_phi=p_lb_phi,
        w_p0=w_p0,
        w_p1=w_p1,
    )


def pac_epsilon(N: int, beta: float) -> float:
    return math.sqrt(max(0.0, 0.5 / float(N) * math.log(1.0 / float(beta))))



# ============================================================
# Rollout / evaluation helpers
# ============================================================

def rollout_episode(env: TenRoomBenchmarkEnv,
                    agent,
                    deterministic: bool,
                    reset_mode: str,
                    terminal_bonus: float = 0.0,
                    store_to_buffer: bool = False,
                    input_noise_cfg: Optional[InputNoiseConfig] = None,
                    rng: Optional[np.random.Generator] = None) -> Dict:
    obs = env.reset(mode=reset_mode)
    total_reward = 0.0
    ep = []
    visited_label_names = [env.initial_label]
    done = False

    while not done:
        curr_state_phys = env.state.copy()
        a, action_info, noisy_obs = agent.select_action(
            obs,
            deterministic=deterministic,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        )
        next_obs, r, done, info = env.step(a)
        next_state_phys = env.state.copy()
        r_aug = r + terminal_bonus if done else r

        if store_to_buffer:
            agent.store_transition(obs, action_info, float(r_aug), next_obs, float(done))

        ep.append({
            "obs": np.asarray(obs, dtype=np.float32).copy(),
            "obs_for_controller": np.asarray(noisy_obs, dtype=np.float32).copy(),
            "state_phys": np.asarray(curr_state_phys, dtype=np.float32).copy(),
            "action": int(a),
            "reward": float(r),
            "reward_aug": float(r_aug),
            "next_obs": np.asarray(next_obs, dtype=np.float32).copy(),
            "next_state_phys": np.asarray(next_state_phys, dtype=np.float32).copy(),
            "label": env.label(curr_state_phys),
            "next_label": info["label"],
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
                           agent,
                           reset_mode: str,
                           store_to_buffer: bool = True,
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
        next_obs, r, done, info = env.step(a)
        if store_to_buffer:
            agent.store_random_transition(obs, a, float(r), next_obs, float(done))

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
                     agent,
                     n_rollouts: int,
                     deterministic: bool,
                     reset_mode: str,
                     terminal_bonus: float = 0.0,
                     store_to_buffer: bool = False,
                     input_noise_cfg: Optional[InputNoiseConfig] = None,
                     seed: Optional[int] = None,
                     enforce_half_split: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []

    if enforce_half_split:
        assert n_rollouts % 2 == 0, "n_rollouts must be even when enforce_half_split=True"
        half = n_rollouts // 2
        modes = ["p0"] * half + ["p1"] * half
    else:
        modes = [reset_mode] * n_rollouts

    for mode in modes:
        out.append(rollout_episode(
            env=env,
            agent=agent,
            deterministic=deterministic,
            reset_mode=mode,
            terminal_bonus=terminal_bonus,
            store_to_buffer=store_to_buffer,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        ))
    return out


def collect_random_rollouts(env: TenRoomBenchmarkEnv,
                            agent,
                            n_rollouts: int,
                            seed: Optional[int] = None,
                            enforce_half_split: bool = False) -> List[Dict]:
    rng = np.random.default_rng(seed)
    out = []

    if enforce_half_split:
        assert n_rollouts % 2 == 0, "n_rollouts must be even when enforce_half_split=True"
        half = n_rollouts // 2
        modes = ["p0"] * half + ["p1"] * half
    else:
        modes = ["mixed"] * n_rollouts

    for mode in modes:
        out.append(rollout_episode_random(env, agent, mode, store_to_buffer=True, rng=rng))
    return out


def warmup_agent(env: TenRoomBenchmarkEnv,
                 agent,
                 train_cfg,
                 train_input_noise_cfg: Optional[InputNoiseConfig] = None) -> None:
    if getattr(agent, "supports_random_warmup", False):
        collect_random_rollouts(
            env=env,
            agent=agent,
            n_rollouts=train_cfg.replay_warmup_episodes,
            seed=train_cfg.seed,
            enforce_half_split=train_cfg.enforce_half_split_train,
        )
    else:
        collect_rollouts(
            env=env,
            agent=agent,
            n_rollouts=train_cfg.replay_warmup_episodes,
            deterministic=False,
            reset_mode=train_cfg.reset_mode_train,
            terminal_bonus=0.0,
            store_to_buffer=True,
            input_noise_cfg=train_input_noise_cfg,
            seed=train_cfg.seed,
            enforce_half_split=train_cfg.enforce_half_split_train,
        )

    for _ in range(train_cfg.warmup_updates):
        agent.update()


def evaluate_policy(env: TenRoomBenchmarkEnv,
                    agent,
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
        store_to_buffer=False,
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
        modes = ["p0"] * (n_eval // 2) + ["p1"] * (n_eval // 2)
    else:
        modes = ["mixed"] * n_eval

    for mode in modes:
        rollouts.append(rollout_episode_with_action_fn(env, action_fn, mode, input_noise_cfg=input_noise_cfg, rng=rng))

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

# ============================================================
# PAC-guided training loop
# ============================================================

def train_pac_guided_td3(env_train: TenRoomBenchmarkEnv,
                             env_eval: TenRoomBenchmarkEnv,
                             agent,
                             train_cfg: PACGuidedTrainConfig,
                             cert_cfg: CertificateNNConfig,
                             device: torch.device,
                             train_input_noise_cfg: Optional[InputNoiseConfig] = None,
                             eval_input_noise_cfg: Optional[InputNoiseConfig] = None,
                             cert_input_noise_cfg: Optional[InputNoiseConfig] = None):
    total_start = time.time()
    history = []

    warmup_agent(env_train, agent, train_cfg, train_input_noise_cfg=train_input_noise_cfg)

    final_result = None
    ema_eval_phi = None
    dual = None

    for k in range(train_cfg.max_outer_iters):
        iter_start = time.time()

        cert_rollouts = collect_rollouts(
            env=env_train,
            agent=agent,
            n_rollouts=train_cfg.N_cert,
            deterministic=True,
            reset_mode=train_cfg.reset_mode_train,
            terminal_bonus=0.0,
            store_to_buffer=False,
            input_noise_cfg=cert_input_noise_cfg,
            seed=train_cfg.seed + 100000 + k,
            enforce_half_split=True,
        )

        dual = solve_dual_certificates(
            cert_rollouts=cert_rollouts,
            horizon_T=train_cfg.horizon_T,
            gamma_bc=train_cfg.gamma_bc,
            nn_cfg=cert_cfg,
            device=device,
        )
        eps = pac_epsilon(train_cfg.N_cert, train_cfg.beta)
        p_lb = dual.p_lb_phi

        cert_phi_rate = float(np.mean([ep["satisfies_phi"] for ep in cert_rollouts]))
        p0_eps = [ep for ep in cert_rollouts if ep["initial_label"] == "p0"]
        p1_eps = [ep for ep in cert_rollouts if ep["initial_label"] == "p1"]
        cert_phi_rate_p0 = float(np.mean([ep["satisfies_phi"] for ep in p0_eps])) if p0_eps else 0.0
        cert_phi_rate_p1 = float(np.mean([ep["satisfies_phi"] for ep in p1_eps])) if p1_eps else 0.0
        n_cert_p0 = len(p0_eps)
        n_cert_p1 = len(p1_eps)

        if (p_lb >= train_cfg.p_min):
            eval_stats = evaluate_policy(
                env_eval,
                agent,
                n_eval=train_cfg.eval_episodes,
                reset_mode=train_cfg.reset_mode_eval,
                input_noise_cfg=eval_input_noise_cfg,
                seed=train_cfg.seed + 200000 + k,
                enforce_half_split=train_cfg.enforce_half_split_eval,
            )
            total_time = time.time() - total_start
            final_result = {
                "outer_iter": k,
                "dual": dual,
                "epsilon": eps,
                "eval": eval_stats,
                "eval_phi_ema": ema_eval_phi,
                "stopped_by_p_lb": True,
                "stopped_by_eval_phi": False,
                "total_time_sec": total_time,
            }
            print(
                "[iter {:04d}] p_lb={:.3f} c1={:.4f} c2={:.4f} ncert_p0={} ncert_p1={} "
                "eval_phi={:.3f} eval_phi_p0={:.3f} eval_phi_p1={:.3f} avg_ret={:.3f}".format(
                    k,
                    p_lb,
                    dual.cert_01.c_bc,
                    dual.cert_10.c_bc,
                    n_cert_p0,
                    n_cert_p1,
                    eval_stats["phi_rate"],
                    eval_stats["phi_rate_p0"],
                    eval_stats["phi_rate_p1"],
                    eval_stats["avg_return"],
                )
            )
            break

        terminal_bonus = -(train_cfg.lambda_bc1 * dual.cert_01.c_bc + train_cfg.lambda_bc2 * dual.cert_10.c_bc)
        collect_rollouts(
            env=env_train,
            agent=agent,
            n_rollouts=train_cfg.episodes_per_outer_iter,
            deterministic=False,
            reset_mode=train_cfg.reset_mode_train,
            terminal_bonus=terminal_bonus,
            store_to_buffer=True,
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
                "p_lb": p_lb,
                "epsilon": eps,
                "c1": dual.cert_01.c_bc,
                "c2": dual.cert_10.c_bc,
                "u1": dual.cert_01.u_local,
                "u2": dual.cert_10.u_local,
                "w_p0": dual.w_p0,
                "w_p1": dual.w_p1,
                "n_cert_p0": n_cert_p0,
                "n_cert_p1": n_cert_p1,
                "cert_phi_rate": cert_phi_rate,
                "cert_phi_rate_p0": cert_phi_rate_p0,
                "cert_phi_rate_p1": cert_phi_rate_p1,
                "npre1": dual.cert_01.n_pre,
                "npost1": dual.cert_01.n_post,
                "npre2": dual.cert_10.n_pre,
                "npost2": dual.cert_10.n_post,
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
                "[iter {:04d}] p_lb={:.3f} c1={:.4f} c2={:.4f} ncert_p0={} ncert_p1={} "
                "npre1={} npost1={} npre2={} npost2={} cert_phi={:.3f} cert_phi_p0={:.3f} cert_phi_p1={:.3f} "
                "train_phi={:.3f} train_phi_p0={:.3f} train_phi_p1={:.3f} train_ret={:.3f} "
                "eval_phi={:.3f} eval_phi_p0={:.3f} eval_phi_p1={:.3f} eval_phi_ema={:.3f} avg_ret={:.3f} "
                "iter_time={:.2f}s total_time={:.2f}s status1={} status2={}".format(
                    k,
                    p_lb,
                    dual.cert_01.c_bc,
                    dual.cert_10.c_bc,
                    n_cert_p0,
                    n_cert_p1,
                    dual.cert_01.n_pre,
                    dual.cert_01.n_post,
                    dual.cert_10.n_pre,
                    dual.cert_10.n_post,
                    cert_phi_rate,
                    cert_phi_rate_p0,
                    cert_phi_rate_p1,
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
                    dual.cert_01.status,
                    dual.cert_10.status,
                )
            )

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
            "dual": dual,
            "epsilon": pac_epsilon(train_cfg.N_cert, train_cfg.beta),
            "eval": eval_stats,
            "eval_phi_ema": ema_eval_phi,
            "stopped_by_p_lb": False,
            "stopped_by_eval_phi": False,
            "total_time_sec": total_time,
        }

    return history, final_result


def robustness_sweep(env_eval: TenRoomBenchmarkEnv,
                     agent,
                     noise_type: str,
                     scales: List[float],
                     n_eval: int = 50,
                     reset_mode: str = "mixed",
                     enforce_half_split: bool = True,
                     clip_min: Optional[float] = None,
                     clip_max: Optional[float] = None,
                     seed: int = 12345) -> List[Dict[str, float]]:
    results = []

    for i, scale in enumerate(scales):
        scale = float(scale)
        if scale <= 0.0:
            noise_cfg = None
        elif noise_type == "uniform":
            noise_cfg = make_uniform_input_noise(scale=scale, clip_min=clip_min, clip_max=clip_max)
        elif noise_type == "gaussian":
            noise_cfg = make_gaussian_input_noise(scale=scale, clip_min=clip_min, clip_max=clip_max)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        stats = evaluate_policy(
            env=env_eval,
            agent=agent,
            n_eval=n_eval,
            reset_mode=reset_mode,
            input_noise_cfg=noise_cfg,
            seed=seed + i,
            enforce_half_split=enforce_half_split,
        )
        results.append({
            "scale": scale,
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
        seed=300,
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

    td3_cfg = TD3Config(
        gamma_rl=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=128,
        exploration_std=0.10,
        exploration_clip=0.25,
        target_policy_noise=0.10,
        target_noise_clip=0.25,
        policy_delay=2,
        gradient_steps_per_iter=50,
        replay_capacity=200000,
    )

    pac_cfg = PACGuidedTrainConfig(
        seed=7,
        max_outer_iters=500,
        replay_warmup_episodes=64,
        warmup_updates=200,
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
        N_cert=100,
        beta=0.05,
        p_min=0.90,
        gamma_bc=0.01,
        horizon_T=14,
        lambda_bc1=25.0,
        lambda_bc2=25.0,
    )

    cert_cfg = CertificateNNConfig(
        hidden_dim=128,
        lr=1e-3,
        epochs=8000,
        lambda_nonneg=5.0,
        lambda_pre=5.0,
        lambda_post=5.0,
        lambda_dyn=5.0,
        lambda_c=3.0,
        tol=1e-3,
        tol_c=1e-3,
    )

    agent = TD3Agent(
        state_dim=env_train.obs_dim,
        action_dim=env_train.action_dim,
        cfg=td3_cfg,
        device=device,
    )

    train_input_noise_cfg = make_uniform_input_noise(scale=0.05)
    eval_input_noise_cfg = make_uniform_input_noise(scale=0.05)
    cert_input_noise_cfg = make_uniform_input_noise(scale=0.05)

    heuristic_stats = evaluate_heuristic_policy(
        env_eval,
        n_eval=200,
        input_noise_cfg=eval_input_noise_cfg,
        seed=2026,
        enforce_half_split=True,
    )
    print("==== HEURISTIC POLICY SANITY CHECK ====")
    print(f"heuristic_phi_rate: {heuristic_stats['phi_rate']:.4f}")
    print(f"heuristic_phi_rate_p0: {heuristic_stats['phi_rate_p0']:.4f}")
    print(f"heuristic_phi_rate_p1: {heuristic_stats['phi_rate_p1']:.4f}")
    print(f"heuristic_avg_return: {heuristic_stats['avg_return']:.4f}")
    print()

    history, res = train_pac_guided_td3(
        env_train=env_train,
        env_eval=env_eval,
        agent=agent,
        train_cfg=pac_cfg,
        cert_cfg=cert_cfg,
        device=device,
        train_input_noise_cfg=train_input_noise_cfg,
        eval_input_noise_cfg=eval_input_noise_cfg,
        cert_input_noise_cfg=cert_input_noise_cfg,
    )

    dual = res.get("dual")
    print("\n==== FINAL RESULT ====")
    print(f"Stopped by p_lb? {res.get('stopped_by_p_lb', False)}")
    print(f"Stopped by eval_phi? {res['stopped_by_eval_phi']}")
    print(f"Outer iter: {res['outer_iter']}")
    if dual is not None:
        print(f"p_lb: {dual.p_lb_phi:.4f}")
        print(f"c1: {dual.cert_01.c_bc:.4f}")
        print(f"c2: {dual.cert_10.c_bc:.4f}")
        print(f"npre1: {dual.cert_01.n_pre}")
        print(f"npost1: {dual.cert_01.n_post}")
        print(f"npre2: {dual.cert_10.n_pre}")
        print(f"npost2: {dual.cert_10.n_post}")
        print(f"status1: {dual.cert_01.status}")
        print(f"status2: {dual.cert_10.status}")
    print(f"epsilon(N,beta): {res['epsilon']:.4f}")
    print(f"eval_phi_rate: {res['eval']['phi_rate']:.4f}")
    print(f"eval_phi_rate_p0: {res['eval']['phi_rate_p0']:.4f}")
    print(f"eval_phi_rate_p1: {res['eval']['phi_rate_p1']:.4f}")
    print(f"eval_phi_ema: {res['eval_phi_ema']:.4f}" if res['eval_phi_ema'] is not None else "eval_phi_ema: None")
    print(f"avg_return: {res['eval']['avg_return']:.4f}")
    print(f"total_time_sec: {res['total_time_sec']:.2f}")

    print("\n==== ROBUSTNESS SWEEP EXAMPLE ====")
    sweep_scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
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
    uniform_results = robustness_sweep(
        env_eval=env_eval,
        agent=agent,
        noise_type="uniform",
        scales=sweep_scales,
        n_eval=50,
        reset_mode=pac_cfg.reset_mode_eval,
        enforce_half_split=pac_cfg.enforce_half_split_eval,
        seed=5252,
    )
    gaussian_results = robustness_sweep(
        env_eval=env_eval,
        agent=agent,
        noise_type="gaussian",
        scales=sweep_scales,
        n_eval=50,
        reset_mode=pac_cfg.reset_mode_eval,
        enforce_half_split=pac_cfg.enforce_half_split_eval,
        seed=6262,
    )
    print("Uniform noise:", uniform_results)
    print("Gaussian noise:", gaussian_results)


if __name__ == "__main__":
    main()
