import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class InputNoiseConfig:
    """
    Noise added to the controller decision input only:
        \tilde s_t = s_t + eta_t
    """
    noise_type: str = "none"   # "none" | "uniform" | "gaussian"
    scale: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    enabled: bool = False

    def is_active(self):
        return self.enabled and self.noise_type != "none" and self.scale > 0.0


def apply_input_noise(obs, noise_cfg=None, rng=None):
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


def make_uniform_input_noise(scale, clip_min=None, clip_max=None):
    return InputNoiseConfig(
        noise_type="uniform",
        scale=scale,
        clip_min=clip_min,
        clip_max=clip_max,
        enabled=True,
    )


def make_gaussian_input_noise(scale, clip_min=None, clip_max=None):
    return InputNoiseConfig(
        noise_type="gaussian",
        scale=scale,
        clip_min=clip_min,
        clip_max=clip_max,
        enabled=True,
    )


# ============================================================
# Truck environment with trailer + bad-prefix DFA (T=2)
# Two certificates are defined by state predicates:
#   A := ma & P0 & P1
#   B := mb & P0 & P2
#   X := not P0
# where
#   P0: operating region,
#   P1: v1 <= va,
#   P2: v1 <= vb.
# ============================================================

class TruckSafetyEnv(object):
    def __init__(
            self,
            dt: float = 0.4,
            max_steps: int = 200,
            seed: int = 7,
            switch_prob: float = 0.05,
            process_noise_std: float = 0.0,
    ):
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.base_seed = int(seed)
        self.reset_count = 0
        self.switch_prob = float(switch_prob)
        self.process_noise_std = float(process_noise_std)

        # Physical parameters from the paper
        self.ks = 4500.0
        self.kd = 4600.0
        self.m = 1000.0

        # Speed limits
        self.va = 15.6
        self.vb = 24.5

        # Operating-region bounds P0 used in the truck-with-trailer experiment
        self.d_abs_max = 0.5
        self.v_min = 5.0
        self.v_max = 35.0
        self.u_min = -4.0
        self.u_max = 4.0

        self.goal_reward = 20.0
        self.fail_penalty = -20.0
        self.safe_step_reward = 0.2
        self.safe_time_bonus_scale = 0.05
        # =========================================

        # Continuous state: [d, v2, v1]
        self.x_dim = 3
        self.mode_dim = 2
        self.num_q = 9
        self.state_dim = self.x_dim + self.mode_dim + self.num_q
        self.action_dim = 1
        self.action_low = np.array([self.u_min], dtype=np.float32)
        self.action_high = np.array([self.u_max], dtype=np.float32)
        self.action_scale = float(self.u_max)

        # Exact ZOH discretization
        Ac = np.array([
            [0.0, -1.0, 1.0],
            [self.ks / self.m, -self.kd / self.m, self.kd / self.m],
            [0.0, 0.0, 0.0],
        ], dtype=np.float64)
        Bc = np.array([[0.0], [0.0], [1.0]], dtype=np.float64)
        Cc = np.eye(3, dtype=np.float64)
        Dc = np.zeros((3, 1), dtype=np.float64)
        Ad, Bd, _, _, _ = signal.cont2discrete((Ac, Bc, Cc, Dc), self.dt, method="zoh")
        self.Ad = np.asarray(Ad, dtype=np.float32)
        self.Bd = np.asarray(Bd, dtype=np.float32)

        self.q_names = [f"q{i}" for i in range(9)]
        self.accepting_states = {3, 7, 8}

        self.x = np.zeros(3, dtype=np.float32)
        self.mode = 0   # 0=a, 1=b
        self.q = 0
        self.t = 0

    def _rng(self):
        return np.random.default_rng(self.base_seed + self.reset_count)

    def mode_onehot(self, mode: int):
        z = np.zeros(self.mode_dim, dtype=np.float32)
        z[int(mode)] = 1.0
        return z

    def q_onehot(self, q: int):
        z = np.zeros(self.num_q, dtype=np.float32)
        z[int(q)] = 1.0
        return z

    def _obs_from_parts(self, x: np.ndarray, mode: int, q: int):
        return np.concatenate([
            x.astype(np.float32),
            self.mode_onehot(mode),
            self.q_onehot(q),
        ], axis=0).astype(np.float32)

    def _current_obs(self):
        return self._obs_from_parts(self.x, self.mode, self.q)

    def _in_P0(self, x: np.ndarray):
        d, v2, v1 = float(x[0]), float(x[1]), float(x[2])
        return (
                abs(d) <= self.d_abs_max and
                self.v_min <= v2 <= self.v_max and
                self.v_min <= v1 <= self.v_max
        )

    def _predicates(self, x: np.ndarray, mode: int):
        p0 = self._in_P0(x)
        p1 = bool(float(x[2]) <= self.va)
        p2 = bool(float(x[2]) <= self.vb)
        ma = bool(mode == 0)
        mb = bool(mode == 1)
        return {
            "P0": p0,
            "P1": p1,
            "P2": p2,
            "ma": ma,
            "mb": mb,
        }

    def _dfa_step(self, q: int, pred: Dict[str, bool]):
        # Accepting states are sinks
        if q in self.accepting_states:
            return q

        p0 = pred["P0"]
        ma = pred["ma"]
        mb = pred["mb"]
        p1 = pred["P1"]
        p2 = pred["P2"]

        if not p0:
            return 8

        if ma:
            if p1:
                return 0
            # violation under mode a
            if q in {0, 4, 5, 6}:
                return 1
            if q == 1:
                return 2
            if q == 2:
                return 3
            return 1

        if mb:
            if p2:
                return 4
            # violation under mode b
            if q in {4, 0, 1, 2}:
                return 5
            if q == 5:
                return 6
            if q == 6:
                return 7
            return 5

        return 8

    def label_from_parts(self, x: np.ndarray, mode: int, q: int):
        pred = self._predicates(x, mode)
        fail = bool(q in self.accepting_states)
        safe = bool(not fail)
        label = {
            "A": bool(pred["ma"] and pred["P0"] and pred["P1"]),
            "B": bool(pred["mb"] and pred["P0"] and pred["P2"]),
            "X": bool(not pred["P0"]),
            "barA": bool(pred["ma"] and pred["P0"] and (not pred["P1"])),
            "barB": bool(pred["mb"] and pred["P0"] and (not pred["P2"])),
            "P0": bool(pred["P0"]),
            "P1": bool(pred["P1"]),
            "P2": bool(pred["P2"]),
            "ma": bool(pred["ma"]),
            "mb": bool(pred["mb"]),
            "fail": fail,
            "safe": safe,
            "q_idx": int(q),
            "q_name": self.q_names[int(q)],
        }
        return label

    def label(self, obs, terminated=False, truncated=False):
        obs = np.asarray(obs, dtype=np.float32)
        x = obs[:self.x_dim]
        mode = int(np.argmax(obs[self.x_dim:self.x_dim + self.mode_dim]))
        q = int(np.argmax(obs[self.x_dim + self.mode_dim:]))
        label = self.label_from_parts(x, mode, q)
        label["goal"] = bool((not label["fail"]) and truncated)
        return label

    def reset(self):
        rng = self._rng()
        self.reset_count += 1
        self.t = 0

        self.mode = int(rng.integers(0, 2))
        target_v = self.va if self.mode == 0 else self.vb

        # Start near a compliant equilibrium-like region
        d0 = float(rng.uniform(-0.05, 0.05))
        v1_0 = float(rng.uniform(target_v - 1.0, target_v - 0.1))
        v2_0 = float(np.clip(v1_0 + rng.uniform(-0.5, 0.5), self.v_min + 0.1, self.v_max - 0.1))
        self.x = np.array([d0, v2_0, v1_0], dtype=np.float32)

        pred0 = self._predicates(self.x, self.mode)
        if pred0["ma"]:
            self.q = 0 if pred0["P1"] and pred0["P0"] else 1
        else:
            self.q = 4 if pred0["P2"] and pred0["P0"] else 5
        if not pred0["P0"]:
            self.q = 8

        return self._current_obs()

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        u = float(np.clip(a[0], self.u_min, self.u_max))

        x_next = self.Ad @ self.x + self.Bd[:, 0] * u
        if self.process_noise_std > 0.0:
            x_next = x_next + np.random.normal(0.0, self.process_noise_std, size=x_next.shape).astype(np.float32)
        x_next = np.asarray(x_next, dtype=np.float32)

        # Exogenous speed-limit switch
        if np.random.rand() < self.switch_prob:
            mode_next = 1 - self.mode
        else:
            mode_next = self.mode

        pred_next = self._predicates(x_next, mode_next)
        q_next = self._dfa_step(self.q, pred_next)

        self.x = x_next
        self.mode = mode_next
        self.q = q_next
        self.t += 1

        obs_next = self._current_obs()
        done = bool((self.q in self.accepting_states) or (self.t >= self.max_steps))
        truncated = bool((self.t >= self.max_steps) and (self.q not in self.accepting_states))
        terminated = bool(self.q in self.accepting_states)

        # =====================================================
        # =====================================================
        if terminated:
            reward = self.fail_penalty
        elif truncated:
            reward = self.goal_reward
        else:
            time_bonus = self.safe_time_bonus_scale * (self.t / self.max_steps)
            reward = self.safe_step_reward + time_bonus

        info = {
            "label": {**self.label_from_parts(x_next, mode_next, q_next),
                      "goal": bool((q_next not in self.accepting_states) and truncated)},
            "terminated_by_fail": bool(self.q in self.accepting_states),
            "terminated_by_goal": bool(truncated and (self.q not in self.accepting_states)),
            "terminated": terminated,
            "truncated": truncated,
            "mode": "a" if self.mode == 0 else "b",
            "raw_action": u,
        }
        return obs_next, float(reward), done, info


# ============================================================
# Replay buffer
# ============================================================

# ============================================================
# Continuous-action SAC
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
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden=256,
                 log_std_min=-20.0, log_std_max=2.0):
        super().__init__()
        low = np.asarray(action_low, dtype=np.float32).reshape(1, action_dim)
        high = np.asarray(action_high, dtype=np.float32).reshape(1, action_dim)
        self.register_buffer("action_low", torch.tensor(low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(high, dtype=torch.float32))
        self.register_buffer("action_mid", torch.tensor((low + high) / 2.0, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, x, deterministic=False):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = mean if deterministic else normal.rsample()
        y = torch.tanh(z)
        action = self.action_mid + self.action_scale * y
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1.0 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = torch.max(torch.min(action, self.action_high), self.action_low)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


@dataclass
class SACConfig:
    gamma_rl: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128
    gradient_steps_per_iter: int = 50
    replay_capacity: int = 200000
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    target_entropy: Optional[float] = None


class SACAgent(object):
    def __init__(self, state_dim, action_dim, action_low, action_high, cfg, device):
        self.cfg = cfg
        self.device = device
        self.action_dim = int(action_dim)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)

        self.actor = Actor(
            state_dim,
            action_dim,
            self.action_low,
            self.action_high,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
        ).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic1_targ = Critic(state_dim, action_dim).to(device)
        self.critic1_targ.load_state_dict(self.critic1.state_dict())
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic2_targ = Critic(state_dim, action_dim).to(device)
        self.critic2_targ.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr)

        self.log_alpha = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)
        self.target_entropy = float(-action_dim if cfg.target_entropy is None else cfg.target_entropy)

        self.replay = ReplayBuffer(cfg.replay_capacity, state_dim, action_dim)

    def select_action(self, state, deterministic=False, input_noise_cfg=None, rng=None):
        noisy_state = apply_input_noise(state, input_noise_cfg, rng)
        s = torch.tensor(noisy_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(s, deterministic=deterministic)
        action = action.squeeze(0).cpu().numpy().astype(np.float32)
        action = np.clip(action, self.action_low, self.action_high).astype(np.float32)
        return action, action.copy(), noisy_state

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return
        batch = self.replay.sample(self.cfg.batch_size, self.device)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_a, next_log_prob = self.actor.sample(batch["next_state"], deterministic=False)
            target_q1 = self.critic1_targ(batch["next_state"], next_a)
            target_q2 = self.critic2_targ(batch["next_state"], next_a)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            y = batch["reward"] + self.cfg.gamma_rl * (1.0 - batch["done"]) * target_q

        q1 = self.critic1(batch["state"], batch["action"])
        q2 = self.critic2(batch["state"], batch["action"])
        critic1_loss = F.mse_loss(q1, y)
        critic2_loss = F.mse_loss(q2, y)

        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        new_a, log_prob = self.actor.sample(batch["state"], deterministic=False)
        q_new = torch.min(self.critic1(batch["state"], new_a), self.critic2(batch["state"], new_a))
        actor_loss = (alpha.detach() * log_prob - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update(self.critic1, self.critic1_targ)
        self.soft_update(self.critic2, self.critic2_targ)

    def soft_update(self, src, tgt):
        tau = self.cfg.tau
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

# ============================================================
# Neural certificate synthesis
# ============================================================

@dataclass
class CertificateNNConfig:
    hidden_dim: int = 128
    lr: float = 1e-3
    epochs: int = 400
    batch_size: int = 256
    lambda_nonneg: float = 10.0
    lambda_pre: float = 10.0
    lambda_post: float = 10.0
    lambda_dyn: float = 10.0
    lambda_c: float = 1.0
    tol: float = 1e-3
    tol_c: float = 1e-3


class CertificateNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_c = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def c_bc(self):
        return F.softplus(self.raw_c)


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
    pre_key: str = ""
    post_key: str = ""


@dataclass
class TwoCertificateResult:
    cert_A: CertificateResult
    cert_B: CertificateResult
    risk_bound: float
    p_lb: float
    c_total: float
    status: str


def extract_sampled_pre_post_sets(cert_rollouts, pre_key: str, post_key: str):
    pre_states = []
    post_states = []
    transitions = []
    all_states = []

    for ep in cert_rollouts:
        seq = ep["transitions"]
        if len(seq) == 0:
            continue

        for tr in seq:
            s = tr["state"]
            ns = tr["next_state"]
            lbl = tr["label"]

            all_states.append(s)
            if bool(lbl.get(post_key, False)):
                post_states.append(s)
            if bool(lbl.get(pre_key, False)):
                pre_states.append(s)
            transitions.append((s, ns))

        final_state = seq[-1]["next_state"]
        final_label = seq[-1]["next_label"]
        all_states.append(final_state)
        if bool(final_label.get(post_key, False)):
            post_states.append(final_state)
        if bool(final_label.get(pre_key, False)):
            pre_states.append(final_state)
        transitions.append((final_state, final_state))

    if len(all_states) == 0:
        return None, None, None, None

    X_all = np.asarray(all_states, dtype=np.float32)
    X_pre = np.asarray(pre_states, dtype=np.float32) if len(pre_states) > 0 else np.zeros((0, X_all.shape[1]), dtype=np.float32)
    X_post = np.asarray(post_states, dtype=np.float32) if len(post_states) > 0 else np.zeros((0, X_all.shape[1]), dtype=np.float32)
    return X_all, X_pre, X_post, transitions


def dedup_rows(X, decimals=6):
    if X is None or X.shape[0] == 0:
        return X
    Xr = np.round(X.astype(np.float64), decimals=decimals)
    _, idx = np.unique(Xr, axis=0, return_index=True)
    idx = np.sort(idx)
    return X[idx]


def solve_sampled_certificate_nn(cert_rollouts,
                                 horizon_T,
                                 gamma_bc,
                                 nn_cfg,
                                 device,
                                 pre_key: str,
                                 post_key: str):
    extracted = extract_sampled_pre_post_sets(cert_rollouts, pre_key=pre_key, post_key=post_key)
    if extracted[0] is None:
        return CertificateResult(
            feasible=False, model_state=None, c_bc=1.0, gamma_bc=gamma_bc,
            T_bc=horizon_T, u_local=1.0, p_lb=0.0, n_pre=0, n_post=0,
            n_trans=0, loss_nonneg=float("inf"), loss_pre=float("inf"),
            loss_post=float("inf"), loss_dyn=float("inf"), loss_obj=1.0,
            max_violation=float("inf"), epochs_used=0, status="no-data",
            pre_key=pre_key, post_key=post_key,
        )

    X_all, X_pre, X_post, transitions = extracted
    X_all = dedup_rows(X_all)
    X_pre = dedup_rows(X_pre)
    X_post = dedup_rows(X_post)

    if len(transitions) == 0 or X_pre.shape[0] == 0 or X_post.shape[0] == 0:
        return CertificateResult(
            feasible=False, model_state=None, c_bc=1.0, gamma_bc=gamma_bc,
            T_bc=horizon_T, u_local=1.0, p_lb=0.0,
            n_pre=int(X_pre.shape[0]), n_post=int(X_post.shape[0]),
            n_trans=len(transitions), loss_nonneg=float("inf"), loss_pre=float("inf"),
            loss_post=float("inf"), loss_dyn=float("inf"), loss_obj=1.0,
            max_violation=float("inf"), epochs_used=0,
            status="insufficient-pre-or-post",
            pre_key=pre_key, post_key=post_key,
        )

    trans_arr = []
    for s, ns in transitions:
        trans_arr.append(np.concatenate([s, ns], axis=0))
    trans_arr = np.asarray(trans_arr, dtype=np.float32)
    trans_arr = dedup_rows(trans_arr)
    state_dim = X_all.shape[1]
    transitions = [(row[:state_dim], row[state_dim:]) for row in trans_arr]

    model = CertificateNet(state_dim=state_dim, hidden_dim=nn_cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=nn_cfg.lr)

    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    X_pre_t = torch.tensor(X_pre, dtype=torch.float32, device=device)
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
        eta_pre = model(X_pre_t)
        eta_post = model(X_post_t)
        eta_s = model(X_s_t)
        eta_ns = model(X_ns_t)
        c_bc_t = model.c_bc()

        loss_nonneg = F.relu(-eta_all).mean()
        loss_pre = F.relu(eta_pre - gamma_bc).mean()
        loss_post = F.relu(1.0 - eta_post).mean()
        loss_dyn = F.relu(eta_ns - eta_s - c_bc_t).mean()
        loss_obj = c_bc_t

        loss = (
                nn_cfg.lambda_nonneg * loss_nonneg +
                nn_cfg.lambda_pre * loss_pre +
                nn_cfg.lambda_post * loss_post +
                nn_cfg.lambda_dyn * loss_dyn +
                nn_cfg.lambda_c * loss_obj
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            eta_all_chk = model(X_all_t)
            eta_pre_chk = model(X_pre_t)
            eta_post_chk = model(X_post_t)
            eta_s_chk = model(X_s_t)
            eta_ns_chk = model(X_ns_t)
            c_bc_chk = model.c_bc()

            final_loss_nonneg = float(F.relu(-eta_all_chk).mean().cpu().item())
            final_loss_pre = float(F.relu(eta_pre_chk - gamma_bc).mean().cpu().item())
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

    with torch.no_grad():
        c_bc = float(model.c_bc().item())

    constraints_satisfied = (
            final_loss_nonneg <= nn_cfg.tol and
            final_loss_pre <= nn_cfg.tol and
            final_loss_post <= nn_cfg.tol and
            final_loss_dyn <= nn_cfg.tol
    )

    if constraints_satisfied:
        u_local = min(1.0, gamma_bc + c_bc * horizon_T)
        p_lb = max(0.0, 1.0 - u_local)
        status = "constraints-satisfied"
    else:
        u_local = 1.0
        p_lb = 0.0
        status = "epoch-limit-infeasible"

    return CertificateResult(
        feasible=bool(constraints_satisfied),
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
        max_violation=max(final_loss_nonneg, final_loss_pre, final_loss_post, final_loss_dyn),
        epochs_used=epochs_used,
        status=status,
        pre_key=pre_key,
        post_key=post_key,
    )


def solve_two_truck_certificates_nn(cert_rollouts,
                                    horizon_T,
                                    gamma_bc,
                                    nn_cfg,
                                    device):
    cert_A = solve_sampled_certificate_nn(
        cert_rollouts=cert_rollouts,
        horizon_T=horizon_T,
        gamma_bc=gamma_bc,
        nn_cfg=nn_cfg,
        device=device,
        pre_key="A",
        post_key="X",
    )
    cert_B = solve_sampled_certificate_nn(
        cert_rollouts=cert_rollouts,
        horizon_T=horizon_T,
        gamma_bc=gamma_bc,
        nn_cfg=nn_cfg,
        device=device,
        pre_key="B",
        post_key="X",
    )

    # PAC composition for the two certificates corresponding to two bad branches.
    # For this simplified truck setting we use
    #   p_lb = 1 - (u_A + u_B), clipped into [0,1].
    risk_bound = min(1.0, cert_A.u_local + cert_B.u_local)
    p_lb = max(0.0, 1.0 - risk_bound)
    c_total = float(cert_A.c_bc + cert_B.c_bc)
    status = "both-feasible" if (cert_A.feasible and cert_B.feasible) else "some-infeasible"

    return TwoCertificateResult(
        cert_A=cert_A,
        cert_B=cert_B,
        risk_bound=risk_bound,
        p_lb=p_lb,
        c_total=c_total,
        status=status,
    )


# ============================================================
# Rollout / evaluation
# ============================================================


def rollout_episode(env,
                    agent,
                    deterministic,
                    terminal_bonus=0.0,
                    store_to_replay=False,
                    input_noise_cfg=None,
                    rng=None):
    s = env.reset()
    total_reward = 0.0
    hit_fail = False
    reached_goal = False
    ep = []

    done = False
    while not done:
        a, a_store, noisy_state = agent.select_action(
            s,
            deterministic=deterministic,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        )
        current_label = env.label(s, terminated=False, truncated=False)
        ns, r, done, info = env.step(a)
        r_aug = r + terminal_bonus if done else r

        if store_to_replay:
            if hasattr(agent, "store_transition"):
                agent.store_transition(s, a_store, float(r_aug), ns, float(done))
            else:
                agent.replay.add(s, a_store, float(r_aug), ns, float(done))
        ep.append({
            "state": np.asarray(s, dtype=np.float32).copy(),
            "state_for_controller": np.asarray(noisy_state, dtype=np.float32).copy(),
            "action": np.asarray(a, dtype=np.float32).copy(),
            "reward": float(r),
            "reward_aug": float(r_aug),
            "next_state": np.asarray(ns, dtype=np.float32).copy(),
            "label": current_label,
            "next_label": info["label"],
        })

        total_reward += r
        s = ns
        hit_fail = hit_fail or info["terminated_by_fail"]
        reached_goal = reached_goal or info["terminated_by_goal"]

    return {
        "transitions": ep,
        "total_reward": total_reward,
        "hit_fail": hit_fail,
        "reached_goal": reached_goal,
        "safe_trace": float(not hit_fail),
        "A_visited": float(any(tr["label"].get("A", False) for tr in ep)),
        "B_visited": float(any(tr["label"].get("B", False) for tr in ep)),
    }


def collect_rollouts(env,
                     agent,
                     n_rollouts,
                     deterministic=True,
                     terminal_bonus=0.0,
                     store_to_replay=False,
                     input_noise_cfg=None,
                     seed=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rollouts):
        out.append(rollout_episode(
            env=env,
            agent=agent,
            deterministic=deterministic,
            terminal_bonus=terminal_bonus,
            store_to_replay=store_to_replay,
            input_noise_cfg=input_noise_cfg,
            rng=rng,
        ))
    return out


def evaluate_policy(env,
                    agent,
                    n_eval=100,
                    input_noise_cfg=None,
                    seed=None):
    rollouts = collect_rollouts(
        env=env,
        agent=agent,
        n_rollouts=n_eval,
        deterministic=True,
        terminal_bonus=0.0,
        store_to_replay=False,
        input_noise_cfg=input_noise_cfg,
        seed=seed,
    )

    safe = [ep["safe_trace"] for ep in rollouts]
    goal = [float(ep["reached_goal"]) for ep in rollouts]
    fail = [float(ep["hit_fail"]) for ep in rollouts]
    returns = [ep["total_reward"] for ep in rollouts]
    a_visit = [ep["A_visited"] for ep in rollouts]
    b_visit = [ep["B_visited"] for ep in rollouts]

    return {
        "safe_rate": float(np.mean(safe)),
        "goal_rate": float(np.mean(goal)),
        "fail_rate": float(np.mean(fail)),
        "avg_return": float(np.mean(returns)),
        "A_visit_rate": float(np.mean(a_visit)),
        "B_visit_rate": float(np.mean(b_visit)),
    }


def robustness_sweep(env,
                     agent,
                     noise_type,
                     scales,
                     n_eval=100):
    out = []
    for scale in scales:
        if noise_type == "uniform":
            noise_cfg = make_uniform_input_noise(scale=scale)
        elif noise_type == "gaussian":
            noise_cfg = make_gaussian_input_noise(scale=scale)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        stats = evaluate_policy(
            env=env,
            agent=agent,
            n_eval=n_eval,
            input_noise_cfg=noise_cfg,
            seed=1234,
        )
        out.append({
            "noise_type": noise_type,
            "scale": float(scale),
            "safe_rate": stats["safe_rate"],
            "goal_rate": stats["goal_rate"],
            "fail_rate": stats["fail_rate"],
            "avg_return": stats["avg_return"],
        })
    return out


# ============================================================
# PAC outer loop
# ============================================================

@dataclass
class PACTrainConfig:
    seed: int = 7
    max_outer_iters: int = 3000
    N_cert: int = 100
    beta: float = 0.05
    p_min: float = 0.8
    lambda_bc: float = 10.0
    gamma_bc: float = 0.10
    horizon_T: int = 2
    replay_warmup_episodes: int = 10
    episodes_per_outer_iter: int = 8
    test_every: int = 1


def pac_epsilon(N, beta):
    return math.sqrt(max(0.0, 0.5 / float(N) * math.log(1.0 / float(beta))))


def train_pac_guided_sac(env_train,
                          env_eval,
                          agent,
                          pac_cfg,
                          cert_nn_cfg,
                          device,
                          cert_input_noise_cfg=None,
                          train_input_noise_cfg=None,
                          eval_input_noise_cfg=None):
    total_start = time.time()
    history = []
    rng = np.random.default_rng(pac_cfg.seed)

    for _ in range(pac_cfg.replay_warmup_episodes):
        rollout_episode(
            env_train,
            agent,
            deterministic=False,
            terminal_bonus=0.0,
            store_to_replay=True,
            input_noise_cfg=train_input_noise_cfg,
            rng=rng,
        )
    for _ in range(200):
        agent.update()

    final_result = None
    last_cert = TwoCertificateResult(
        cert_A=CertificateResult(False, None, 1.0, pac_cfg.gamma_bc, pac_cfg.horizon_T, 1.0, 0.0,
                                 0, 0, 0, float("inf"), float("inf"), float("inf"), float("inf"),
                                 1.0, float("inf"), 0, "init", "A", "X"),
        cert_B=CertificateResult(False, None, 1.0, pac_cfg.gamma_bc, pac_cfg.horizon_T, 1.0, 0.0,
                                 0, 0, 0, float("inf"), float("inf"), float("inf"), float("inf"),
                                 1.0, float("inf"), 0, "init", "B", "X"),
        risk_bound=1.0,
        p_lb=0.0,
        c_total=2.0,
        status="init",
    )

    for k in range(pac_cfg.max_outer_iters):
        iter_start = time.time()

        cert_rollouts = collect_rollouts(
            env=env_train,
            agent=agent,
            n_rollouts=pac_cfg.N_cert,
            deterministic=True,
            terminal_bonus=0.0,
            store_to_replay=False,
            input_noise_cfg=cert_input_noise_cfg,
            seed=pac_cfg.seed + k,
        )

        cert_res = solve_two_truck_certificates_nn(
            cert_rollouts=cert_rollouts,
            horizon_T=pac_cfg.horizon_T,
            gamma_bc=pac_cfg.gamma_bc,
            nn_cfg=cert_nn_cfg,
            device=device,
        )
        last_cert = cert_res

        p_lb = cert_res.p_lb
        eps = pac_epsilon(pac_cfg.N_cert, pac_cfg.beta)

        cert_safe_rate = float(np.mean([ep["safe_trace"] for ep in cert_rollouts]))
        cert_goal_rate = float(np.mean([float(ep["reached_goal"]) for ep in cert_rollouts]))
        cert_fail_rate = float(np.mean([float(ep["hit_fail"]) for ep in cert_rollouts]))

        if cert_safe_rate == 1.0:
            eval_stats = evaluate_policy(
                env_eval,
                agent,
                n_eval=200,
                input_noise_cfg=eval_input_noise_cfg,
                seed=pac_cfg.seed + 100000 + k,
            )
            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            print(
                "[iter {:04d}] p_lb={:.3f} risk={:.3f} c_sum={:.4f} "
                "uA={:.3f} cA={:.4f} npreA={} npostA={} uB={:.3f} cB={:.4f} npreB={} npostB={} "
                "cert_safe={:.3f} cert_goal={:.3f} cert_fail={:.3f} "
                "safe_eval={:.3f} goal_eval={:.3f} fail_eval={:.3f} "
                "iter_time={:.2f}s total_time={:.2f}s".format(
                    k, p_lb, cert_res.risk_bound, cert_res.c_total,
                    cert_res.cert_A.u_local, cert_res.cert_A.c_bc, cert_res.cert_A.n_pre, cert_res.cert_A.n_post,
                    cert_res.cert_B.u_local, cert_res.cert_B.c_bc, cert_res.cert_B.n_pre, cert_res.cert_B.n_post,
                    cert_safe_rate, cert_goal_rate, cert_fail_rate,
                    eval_stats["safe_rate"], eval_stats["goal_rate"], eval_stats["fail_rate"],
                    iter_time, total_time,
                )
            )
            print("[stop] iter={} p_lb={:.3f} >= p_min={:.3f}, epsilon={:.4f}".format(
                k, p_lb, pac_cfg.p_min, eps
            ))
            final_result = {
                "outer_iter": k,
                "certificate": cert_res,
                "epsilon": eps,
                "eval": eval_stats,
                "stopped_by_p_lb": True,
                "total_time_sec": total_time,
            }
            break


        terminal_bonus = -pac_cfg.lambda_bc * cert_res.c_total
        for _ in range(pac_cfg.episodes_per_outer_iter):
            rollout_episode(
                env_train,
                agent,
                deterministic=False,
                terminal_bonus=terminal_bonus,
                store_to_replay=True,
                input_noise_cfg=train_input_noise_cfg,
                rng=rng,
            )
            for _ in range(agent.cfg.gradient_steps_per_iter):
                agent.update()

        if (k % pac_cfg.test_every) == 0 or (k == pac_cfg.max_outer_iters - 1):
            eval_stats = evaluate_policy(
                env_eval,
                agent,
                n_eval=100,
                input_noise_cfg=eval_input_noise_cfg,
                seed=pac_cfg.seed + 200000 + k,
            )
            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            history.append({
                "iter": k,
                "p_lb": p_lb,
                "risk_bound": cert_res.risk_bound,
                "c_total": cert_res.c_total,
                "u_A": cert_res.cert_A.u_local,
                "u_B": cert_res.cert_B.u_local,
                "c_A": cert_res.cert_A.c_bc,
                "c_B": cert_res.cert_B.c_bc,
                "n_pre_A": cert_res.cert_A.n_pre,
                "n_post_A": cert_res.cert_A.n_post,
                "n_pre_B": cert_res.cert_B.n_pre,
                "n_post_B": cert_res.cert_B.n_post,
                "epsilon": eps,
                "cert_safe_rate": cert_safe_rate,
                "cert_goal_rate": cert_goal_rate,
                "cert_fail_rate": cert_fail_rate,
                "safe_eval": eval_stats["safe_rate"],
                "goal_eval": eval_stats["goal_rate"],
                "fail_eval": eval_stats["fail_rate"],
                "avg_return": eval_stats["avg_return"],
                "iter_time": iter_time,
                "total_time": total_time,
            })
            print(
                "[iter {:04d}] p_lb={:.3f} risk={:.3f} c_sum={:.4f} "
                "uA={:.3f} cA={:.4f} uB={:.3f} cB={:.4f} "
                "cert_safe={:.3f} cert_goal={:.3f} cert_fail={:.3f} "
                "safe_eval={:.3f} goal_eval={:.3f} fail_eval={:.3f} avg_ret={:.2f} "
                "iter_time={:.2f}s total_time={:.2f}s status={}".format(
                    k, p_lb, cert_res.risk_bound, cert_res.c_total,
                    cert_res.cert_A.u_local, cert_res.cert_A.c_bc,
                    cert_res.cert_B.u_local, cert_res.cert_B.c_bc,
                    cert_safe_rate, cert_goal_rate, cert_fail_rate,
                    eval_stats["safe_rate"], eval_stats["goal_rate"], eval_stats["fail_rate"],
                    eval_stats["avg_return"], iter_time, total_time, cert_res.status
                )
            )

    if final_result is None:
        total_time = time.time() - total_start
        eval_stats = evaluate_policy(
            env_eval,
            agent,
            n_eval=200,
            input_noise_cfg=eval_input_noise_cfg,
            seed=pac_cfg.seed + 300000,
        )
        final_result = {
            "outer_iter": pac_cfg.max_outer_iters - 1,
            "certificate": last_cert,
            "epsilon": pac_epsilon(pac_cfg.N_cert, pac_cfg.beta),
            "eval": eval_stats,
            "stopped_by_p_lb": False,
            "total_time_sec": total_time,
        }

    return history, final_result


# ============================================================
# Main
# ============================================================


def main():
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_train = TruckSafetyEnv(dt=0.4, max_steps=100, seed=9, switch_prob=0.005, process_noise_std=0.0)
    env_eval = TruckSafetyEnv(dt=0.4, max_steps=100, seed=123, switch_prob=0.005, process_noise_std=0.0)

    sac_cfg = SACConfig(
        gamma_rl=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        batch_size=128,
        gradient_steps_per_iter=50,
        replay_capacity=200000,
        log_std_min=-20.0,
        log_std_max=2.0,
        target_entropy=None,
    )

    pac_cfg = PACTrainConfig(
        seed=7,
        max_outer_iters=1000,
        N_cert=100,
        beta=0.05,
        p_min=0.95,
        lambda_bc=0,
        gamma_bc=0.0001,
        horizon_T=100,
        replay_warmup_episodes=0,
        episodes_per_outer_iter=8,
        test_every=1,
    )

    cert_nn_cfg = CertificateNNConfig(
        hidden_dim=128,
        lr=5e-3,
        epochs=0,
        batch_size=256,
        lambda_nonneg=10.0,
        lambda_pre=10.0,
        lambda_post=10.0,
        lambda_dyn=10.0,
        lambda_c=1.0,
        tol=1e-3,
        tol_c=1e-4,
    )

    agent = SACAgent(
        state_dim=env_train.state_dim,
        action_dim=env_train.action_dim,
        action_low=env_train.action_low,
        action_high=env_train.action_high,
        cfg=sac_cfg,
        device=device,
    )

    cert_input_noise_cfg = make_gaussian_input_noise(scale=0)
    train_input_noise_cfg = make_gaussian_input_noise(scale=0)
    eval_input_noise_cfg = make_gaussian_input_noise(scale=0)

    history, res = train_pac_guided_sac(
        env_train=env_train,
        env_eval=env_eval,
        agent=agent,
        pac_cfg=pac_cfg,
        cert_nn_cfg=cert_nn_cfg,
        device=device,
        cert_input_noise_cfg=cert_input_noise_cfg,
        train_input_noise_cfg=train_input_noise_cfg,
        eval_input_noise_cfg=eval_input_noise_cfg,
    )

    cert = res["certificate"]
    print("\n==== FINAL RESULT ====")
    print(f"Stopped by p_lb? {res['stopped_by_p_lb']}")
    print(f"Outer iter: {res['outer_iter']}")
    print(f"p_lb_total: {cert.p_lb:.4f}")
    print(f"risk_bound_total: {cert.risk_bound:.4f}")
    print(f"c_total: {cert.c_total:.4f}")

    print("\n---- Certificate A (pre=A, post=X) ----")
    print(f"u_A: {cert.cert_A.u_local:.4f}")
    print(f"c_A: {cert.cert_A.c_bc:.4f}")
    print(f"n_pre_A: {cert.cert_A.n_pre}")
    print(f"n_post_A: {cert.cert_A.n_post}")
    print(f"n_trans_A: {cert.cert_A.n_trans}")
    print(f"loss_nonneg_A: {cert.cert_A.loss_nonneg:.4e}")
    print(f"loss_pre_A: {cert.cert_A.loss_pre:.4e}")
    print(f"loss_post_A: {cert.cert_A.loss_post:.4e}")
    print(f"loss_dyn_A: {cert.cert_A.loss_dyn:.4e}")
    print(f"loss_obj_A: {cert.cert_A.loss_obj:.4e}")
    print(f"epochs_used_A: {cert.cert_A.epochs_used}")
    print(f"status_A: {cert.cert_A.status}")

    print("\n---- Certificate B (pre=B, post=X) ----")
    print(f"u_B: {cert.cert_B.u_local:.4f}")
    print(f"c_B: {cert.cert_B.c_bc:.4f}")
    print(f"n_pre_B: {cert.cert_B.n_pre}")
    print(f"n_post_B: {cert.cert_B.n_post}")
    print(f"n_trans_B: {cert.cert_B.n_trans}")
    print(f"loss_nonneg_B: {cert.cert_B.loss_nonneg:.4e}")
    print(f"loss_pre_B: {cert.cert_B.loss_pre:.4e}")
    print(f"loss_post_B: {cert.cert_B.loss_post:.4e}")
    print(f"loss_dyn_B: {cert.cert_B.loss_dyn:.4e}")
    print(f"loss_obj_B: {cert.cert_B.loss_obj:.4e}")
    print(f"epochs_used_B: {cert.cert_B.epochs_used}")
    print(f"status_B: {cert.cert_B.status}")

    print(f"\nepsilon(N,beta): {res['epsilon']:.4f}")
    print(f"safe_eval: {res['eval']['safe_rate']:.4f}")
    print(f"goal_eval: {res['eval']['goal_rate']:.4f}")
    print(f"fail_eval: {res['eval']['fail_rate']:.4f}")
    print(f"avg_return: {res['eval']['avg_return']:.4f}")
    print(f"total_time_sec: {res['total_time_sec']:.2f}")

    print("\n==== ROBUSTNESS SWEEP EXAMPLE ====")
    uniform_results = robustness_sweep(
        env_eval, agent, "uniform",
        scales=[0.02, 0.08, 0.14, 0.20, 0.26, 0.32, 0.38, 0.40],
        n_eval=50,
    )
    gaussian_results = robustness_sweep(
        env_eval, agent, "gaussian",
        scales=[0.02, 0.08, 0.14, 0.20, 0.26, 0.32, 0.38, 0.40],
        n_eval=50,
    )
    print("Uniform noise:", uniform_results)
    print("Gaussian noise:", gaussian_results)

    return history, res


if __name__ == "__main__":
    main()
