import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym
    GYMNASIUM = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class InputNoiseConfig:
    noise_type: str = "none"
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
        raise ValueError("Unknown noise_type: {}".format(noise_cfg.noise_type))
    x_noisy = x + noise
    if noise_cfg.clip_min is not None or noise_cfg.clip_max is not None:
        lo = -np.inf if noise_cfg.clip_min is None else noise_cfg.clip_min
        hi = np.inf if noise_cfg.clip_max is None else noise_cfg.clip_max
        x_noisy = np.clip(x_noisy, lo, hi)
    return x_noisy.astype(np.float32)


def make_uniform_input_noise(scale, clip_min=None, clip_max=None):
    return InputNoiseConfig("uniform", scale, clip_min, clip_max, True)


def make_gaussian_input_noise(scale, clip_min=None, clip_max=None):
    return InputNoiseConfig("gaussian", scale, clip_min, clip_max, True)


class PendulumSafetyEnv(object):
    def __init__(self,
                 env_name="Pendulum-v1",
                 max_steps=200,
                 seed=7,
                 safe_angle_deg=90.0,
                 unsafe_angle_deg=90.0,
                 safe_speed=1.0,
                 unsafe_speed=2.5,
                 init_angle_deg=10.0,
                 init_speed=0.5):
        self.env = gym.make(env_name)
        self.max_steps = max_steps
        self.base_seed = seed
        self.reset_count = 0
        self.step_count = 0

        self.safe_angle = math.radians(safe_angle_deg)
        self.unsafe_angle = math.radians(unsafe_angle_deg)
        self.safe_speed = float(safe_speed)
        self.unsafe_speed = float(unsafe_speed)

        self.init_angle = math.radians(init_angle_deg)
        self.init_speed = float(init_speed)

        self.state_dim = 3
        self.action_dim = 1
        self.max_action = float(self.env.action_space.high[0])

    @staticmethod
    def obs_to_angle(obs):
        return math.atan2(float(obs[1]), float(obs[0]))

    def reset(self):
        seed = self.base_seed + self.reset_count
        self.reset_count += 1
        self.step_count = 0
        if GYMNASIUM:
            _, _ = self.env.reset(seed=seed)
        else:
            try:
                self.env.seed(seed)
            except Exception:
                pass
            _ = self.env.reset()
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(-self.init_angle, self.init_angle)
        theta_dot0 = rng.uniform(-self.init_speed, self.init_speed)
        self.env.unwrapped.state = np.array([theta0, theta_dot0], dtype=np.float32)
        obs = self.env.unwrapped._get_obs()
        return np.asarray(obs, dtype=np.float32)

    def label(self, obs, terminated=False, truncated=False):
        theta = self.obs_to_angle(obs)
        theta_dot = float(obs[2])

        # Safe region: |theta| <= pi/2
        # Unsafe region: all other states
        in_safe = (abs(theta) <= (math.pi / 2.0))
        in_unsafe = (not in_safe)

        return {
            "fail": bool(in_unsafe),
            "safe": bool(in_safe),
            "unsafe": bool(in_unsafe),
            "middle": False,
            "goal": bool((not in_unsafe) and truncated),
            "theta": float(theta),
            "theta_dot": float(theta_dot),
        }

    def step(self, action):
        self.step_count += 1
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -self.max_action, self.max_action)
        out = self.env.step(a)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        obs = np.asarray(obs, dtype=np.float32)
        label = self.label(obs, terminated=terminated, truncated=truncated)
        unsafe_done = bool(label["fail"])
        time_done = bool(truncated or (self.step_count >= self.max_steps))
        done = bool(unsafe_done or time_done)
        info = dict(info)
        info["label"] = label
        info["terminated_by_fail"] = unsafe_done
        info["terminated_by_goal"] = bool((not unsafe_done) and time_done)
        info["terminated"] = bool(unsafe_done or terminated)
        info["truncated"] = bool(time_done)
        return obs, float(reward), done, info



class PPOBuffer(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)

    def add(self, s, a, r, d, log_prob, value):
        self.states.append(np.asarray(s, dtype=np.float32).copy())
        self.actions.append(np.asarray(a, dtype=np.float32).reshape(-1).copy())
        self.rewards.append(float(r))
        self.dones.append(float(d))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def make_batch(self, gamma_rl, gae_lambda, device):
        n = len(self.states)
        states = torch.tensor(np.asarray(self.states, dtype=np.float32), dtype=torch.float32, device=device)
        actions = torch.tensor(np.asarray(self.actions, dtype=np.float32), dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(np.asarray(self.log_probs, dtype=np.float32), dtype=torch.float32, device=device)

        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_nonterminal = 1.0 - dones[t]
                next_value = 0.0
            else:
                next_nonterminal = 1.0 - dones[t]
                next_value = values[t + 1]
            delta = rewards[t] + gamma_rl * next_value * next_nonterminal - values[t]
            last_gae = delta + gamma_rl * gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std(unbiased=False)
        advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)

        return {
            "state": states,
            "action": actions,
            "old_log_prob": old_log_probs,
            "advantage": advantages_t,
            "return": returns_t,
        }


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden=256, log_std_init=-0.5):
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * float(log_std_init))

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def atanh(x):
        x = torch.clamp(x, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def sample(self, x, deterministic=False):
        mean = self.forward(x)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mean)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = mean if deterministic else normal.sample()
        y = torch.tanh(z)
        action = self.max_action * y
        correction = torch.log(self.max_action * (1.0 - y.pow(2)) + 1e-6)
        log_prob = (normal.log_prob(z) - correction).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, x, actions):
        mean = self.forward(x)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mean)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        scaled = torch.clamp(actions / self.max_action, -0.999999, 0.999999)
        z = self.atanh(scaled)
        y = torch.tanh(z)
        correction = torch.log(self.max_action * (1.0 - y.pow(2)) + 1e-6)
        log_prob = (normal.log_prob(z) - correction).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1).mean()
        return log_prob, entropy


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s):
        return self.net(s).squeeze(-1)


@dataclass
class PPOConfig:
    gamma_rl: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    clip_ratio: float = 0.20
    entropy_coef: float = 0.001
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    gradient_steps_per_iter: int = 50
    log_std_init: float = -0.5


class PPOAgent(object):
    def __init__(self, state_dim, action_dim, max_action, cfg, device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.max_action = float(max_action)

        self.actor = Actor(state_dim, action_dim, max_action, log_std_init=cfg.log_std_init).to(device)
        self.critic = Critic(state_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.buffer = PPOBuffer()
        self._active_batch = None
        self._active_update_steps = 0

    def select_action(self, state, deterministic=False, input_noise_cfg=None, rng=None):
        noisy_state = apply_input_noise(state, input_noise_cfg, rng)
        s = torch.tensor(noisy_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, log_prob_t = self.actor.sample(s, deterministic=deterministic)
            value_t = self.critic(s)
        action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
        action = np.clip(action, -self.max_action, self.max_action).astype(np.float32)
        action_info = {
            "action": action.copy(),
            "log_prob": float(log_prob_t.item()),
            "value": float(value_t.item()),
            "state_for_controller": np.asarray(noisy_state, dtype=np.float32).copy(),
        }
        return action, action_info, noisy_state

    def store_transition(self, s, action_info, r, ns, d):
        del ns
        state_for_update = action_info.get("state_for_controller", s)
        self.buffer.add(
            s=state_for_update,
            a=action_info["action"],
            r=r,
            d=d,
            log_prob=action_info["log_prob"],
            value=action_info["value"],
        )

    def update(self):
        if self._active_batch is None:
            if len(self.buffer) == 0:
                return
            self._active_batch = self.buffer.make_batch(
                gamma_rl=self.cfg.gamma_rl,
                gae_lambda=self.cfg.gae_lambda,
                device=self.device,
            )
            self.buffer.clear()
            self._active_update_steps = 0

        batch = self._active_batch
        n = batch["state"].shape[0]
        if n == 0:
            self._active_batch = None
            return

        mb_size = min(self.cfg.batch_size, n)
        idx = torch.randperm(n, device=self.device)[:mb_size]
        states = batch["state"][idx]
        actions = batch["action"][idx]
        old_log_probs = batch["old_log_prob"][idx]
        advantages = batch["advantage"][idx]
        returns = batch["return"][idx]

        log_probs, entropy = self.actor.evaluate_actions(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * entropy
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_opt.step()

        values = self.critic(states)
        critic_loss = self.cfg.value_coef * F.mse_loss(values, returns)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.critic_opt.step()

        self._active_update_steps += 1
        if self._active_update_steps >= self.cfg.gradient_steps_per_iter:
            self._active_batch = None
            self._active_update_steps = 0

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


def extract_sampled_pre_post_sets(cert_rollouts):
    pre_states, post_states, transitions, all_states = [], [], [], []
    for ep in cert_rollouts:
        seq = ep["transitions"]
        if len(seq) == 0:
            continue
        for tr in seq:
            s = tr["state"]
            ns = tr["next_state"]
            lbl = tr["label"]
            all_states.append(s)
            if bool(lbl["safe"]):
                pre_states.append(s)
            if bool(lbl["fail"]):
                post_states.append(s)
            transitions.append((s, ns))
        final_state = seq[-1]["next_state"]
        final_label = seq[-1]["next_label"]
        all_states.append(final_state)
        if bool(final_label["safe"]):
            pre_states.append(final_state)
        if bool(final_label["fail"]):
            post_states.append(final_state)
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


def solve_sampled_fail_certificate_nn(cert_rollouts, horizon_T, gamma_bc, nn_cfg, device):
    extracted = extract_sampled_pre_post_sets(cert_rollouts)
    if extracted[0] is None:
        return CertificateResult(False, None, 1.0, gamma_bc, horizon_T, 1.0, 0.0, 0, 0, 0,
                                 float("inf"), float("inf"), float("inf"), float("inf"),
                                 1.0, float("inf"), 0, "no-data")
    X_all, X_pre, X_post, transitions = extracted
    X_all = dedup_rows(X_all)
    X_pre = dedup_rows(X_pre)
    X_post = dedup_rows(X_post)
    if len(transitions) == 0:
        return CertificateResult(False, None, 1.0, gamma_bc, horizon_T, 1.0, 0.0,
                                 int(X_pre.shape[0]), int(X_post.shape[0]), 0,
                                 float("inf"), float("inf"), float("inf"), float("inf"),
                                 1.0, float("inf"), 0, "no-transitions")
    trans_arr = np.asarray([np.concatenate([s, ns], axis=0) for s, ns in transitions], dtype=np.float32)
    trans_arr = dedup_rows(trans_arr)
    state_dim = X_all.shape[1]
    transitions = [(row[:state_dim], row[state_dim:]) for row in trans_arr]

    if X_post.shape[0] == 0:
        u_local = gamma_bc
        p_lb = max(0.0, 1.0 - u_local)
        return CertificateResult(True, None, 0.0, gamma_bc, horizon_T, u_local, p_lb,
                                 int(X_pre.shape[0]), 0, len(transitions),
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
                                 "no-unsafe-sample-direct-c-zero")

    model = CertificateNet(state_dim=state_dim, hidden_dim=nn_cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=nn_cfg.lr)
    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    X_pre_t = torch.tensor(X_pre, dtype=torch.float32, device=device) if X_pre.shape[0] > 0 else None
    X_post_t = torch.tensor(X_post, dtype=torch.float32, device=device)
    X_s_t = torch.tensor(np.asarray([s for s, _ in transitions], dtype=np.float32), dtype=torch.float32, device=device)
    X_ns_t = torch.tensor(np.asarray([ns for _, ns in transitions], dtype=np.float32), dtype=torch.float32, device=device)

    final_loss_nonneg = final_loss_pre = final_loss_post = final_loss_dyn = final_loss_obj = float("inf")
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
        loss = (nn_cfg.lambda_nonneg * loss_nonneg + nn_cfg.lambda_pre * loss_pre +
                nn_cfg.lambda_post * loss_post + nn_cfg.lambda_dyn * loss_dyn +
                nn_cfg.lambda_c * loss_obj)
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

    with torch.no_grad():
        c_bc = float(model.c_bc().item())
    constraints_satisfied = (final_loss_nonneg <= nn_cfg.tol and
                             final_loss_pre <= nn_cfg.tol and
                             final_loss_post <= nn_cfg.tol and
                             final_loss_dyn <= nn_cfg.tol)
    if constraints_satisfied:
        u_local = min(1.0, gamma_bc + c_bc * horizon_T)
        p_lb = max(0.0, 1.0 - u_local)
        status = "constraints-satisfied"
    else:
        u_local = 1.0
        p_lb = 0.0
        status = "epoch-limit-infeasible"
    max_violation = max(final_loss_nonneg, final_loss_pre, final_loss_post, final_loss_dyn)
    return CertificateResult(bool(constraints_satisfied),
                             {k: v.detach().cpu() for k, v in model.state_dict().items()},
                             c_bc, gamma_bc, horizon_T, u_local, p_lb,
                             int(X_pre.shape[0]), int(X_post.shape[0]), len(transitions),
                             final_loss_nonneg, final_loss_pre, final_loss_post, final_loss_dyn,
                             final_loss_obj, max_violation, epochs_used, status)


def rollout_episode(env, agent, deterministic, terminal_bonus=0.0, store_to_replay=False,
                    input_noise_cfg=None, rng=None):
    s = env.reset()
    total_reward = 0.0
    hit_fail = False
    reached_goal = False
    ep = []
    done = False
    while not done:
        a, a_store, noisy_state = agent.select_action(s, deterministic, input_noise_cfg, rng)
        ns, r, done, info = env.step(a)
        r_aug = r + terminal_bonus if done else r
        if store_to_replay:
            if hasattr(agent, "store_transition"):
                agent.store_transition(s, a_store, float(r_aug), ns, float(done))
            else:
                agent.replay.add(s, a_store, float(r_aug), ns, float(done))
        current_label = env.label(s, terminated=False, truncated=False)
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
    }


def collect_rollouts(env, agent, n_rollouts, deterministic=True, terminal_bonus=0.0,
                     store_to_replay=False, input_noise_cfg=None, seed=None):
    rng = np.random.default_rng(seed)
    return [rollout_episode(env, agent, deterministic, terminal_bonus, store_to_replay, input_noise_cfg, rng)
            for _ in range(n_rollouts)]


def evaluate_policy(env, agent, n_eval=100, input_noise_cfg=None, seed=None):
    rollouts = collect_rollouts(env, agent, n_eval, True, 0.0, False, input_noise_cfg, seed)
    safe = [ep["safe_trace"] for ep in rollouts]
    goal = [float(ep["reached_goal"]) for ep in rollouts]
    fail = [float(ep["hit_fail"]) for ep in rollouts]
    returns = [ep["total_reward"] for ep in rollouts]
    return {
        "safe_rate": float(np.mean(safe)),
        "goal_rate": float(np.mean(goal)),
        "fail_rate": float(np.mean(fail)),
        "avg_return": float(np.mean(returns)),
    }


def robustness_sweep(env, agent, noise_type, scales, n_eval=100):
    out = []
    for scale in scales:
        if noise_type == "uniform":
            noise_cfg = make_uniform_input_noise(scale=scale)
        elif noise_type == "gaussian":
            noise_cfg = make_gaussian_input_noise(scale=scale)
        else:
            raise ValueError("Unknown noise_type: {}".format(noise_type))
        stats = evaluate_policy(env, agent, n_eval=n_eval, input_noise_cfg=noise_cfg, seed=1234)
        out.append({
            "noise_type": noise_type,
            "scale": float(scale),
            "safe_rate": stats["safe_rate"],
            "goal_rate": stats["goal_rate"],
            "fail_rate": stats["fail_rate"],
            "avg_return": stats["avg_return"],
        })
    return out


@dataclass
class PACTrainConfig:
    seed: int = 7
    max_outer_iters: int = 5000
    N_cert: int = 100
    beta: float = 0.05
    p_min: float = 0.8
    lambda_bc: float = 10.0
    gamma_bc: float = 0.10
    horizon_T: int = 200
    replay_warmup_episodes: int = 50
    episodes_per_outer_iter: int = 8
    test_every: int = 1


def pac_epsilon(N, beta):
    return math.sqrt(max(0.0, 0.5 / float(N) * math.log(1.0 / float(beta))))


def train_pac_guided_ppo(env_train, env_eval, agent, pac_cfg, cert_nn_cfg, device,
                          cert_input_noise_cfg=None, train_input_noise_cfg=None,
                          eval_input_noise_cfg=None):
    total_start = time.time()
    history = []
    rng = np.random.default_rng(pac_cfg.seed)

    for _ in range(pac_cfg.replay_warmup_episodes):
        rollout_episode(env_train, agent, False, 0.0, True, train_input_noise_cfg, rng)
    for _ in range(200):
        agent.update()

    final_result = None
    last_cert = CertificateResult(False, None, 1.0, pac_cfg.gamma_bc, pac_cfg.horizon_T, 1.0, 0.0,
                                  0, 0, 0, float("inf"), float("inf"), float("inf"),
                                  float("inf"), 1.0, float("inf"), 0, "init")

    for k in range(pac_cfg.max_outer_iters):
        iter_start = time.time()
        cert_rollouts = collect_rollouts(env_train, agent, pac_cfg.N_cert, True, 0.0, False,
                                         cert_input_noise_cfg, pac_cfg.seed + k)
        cert_res = solve_sampled_fail_certificate_nn(cert_rollouts, pac_cfg.horizon_T,
                                                     pac_cfg.gamma_bc, cert_nn_cfg, device)
        last_cert = cert_res
        p_lb = cert_res.p_lb
        eps = pac_epsilon(pac_cfg.N_cert, pac_cfg.beta)

        cert_safe_rate = float(np.mean([ep["safe_trace"] for ep in cert_rollouts]))
        cert_goal_rate = float(np.mean([float(ep["reached_goal"]) for ep in cert_rollouts]))
        cert_fail_rate = float(np.mean([float(ep["hit_fail"]) for ep in cert_rollouts]))

        if cert_res.feasible and p_lb >= pac_cfg.p_min:
            eval_stats = evaluate_policy(env_eval, agent, 200, eval_input_noise_cfg, pac_cfg.seed + 100000 + k)
            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            print("[iter {:04d}] p_lb={:.3f} u={:.3f} c={:.4f} "
                  "n_pre={} n_post={} n_trans={} cert_safe={:.3f} cert_goal={:.3f} cert_fail={:.3f} "
                  "safe_eval={:.3f} goal_eval={:.3f} fail_eval={:.3f} "
                  "iter_time={:.2f}s total_time={:.2f}s "
                  "ln={:.2e} lp={:.2e} lq={:.2e} ld={:.2e} lo={:.2e} max_vio={:.2e} ep={:d}".format(
                      k, p_lb, cert_res.u_local, cert_res.c_bc,
                      cert_res.n_pre, cert_res.n_post, cert_res.n_trans,
                      cert_safe_rate, cert_goal_rate, cert_fail_rate,
                      eval_stats["safe_rate"], eval_stats["goal_rate"], eval_stats["fail_rate"],
                      iter_time, total_time,
                      cert_res.loss_nonneg, cert_res.loss_pre, cert_res.loss_post,
                      cert_res.loss_dyn, cert_res.loss_obj,
                      cert_res.max_violation, cert_res.epochs_used))
            print("[stop] iter={} p_lb={:.3f} >= p_min={:.3f}, epsilon={:.4f}".format(
                k, p_lb, pac_cfg.p_min, eps))
            final_result = {
                "outer_iter": k,
                "certificate": cert_res,
                "epsilon": eps,
                "eval": eval_stats,
                "stopped_by_p_lb": True,
                "total_time_sec": total_time,
            }
            break

        terminal_bonus = -pac_cfg.lambda_bc * cert_res.c_bc
        for _ in range(pac_cfg.episodes_per_outer_iter):
            rollout_episode(env_train, agent, False, terminal_bonus, True, train_input_noise_cfg, rng)
            for _ in range(agent.cfg.gradient_steps_per_iter):
                agent.update()

        if (k % pac_cfg.test_every) == 0 or (k == pac_cfg.max_outer_iters - 1):
            eval_stats = evaluate_policy(env_eval, agent, 100, eval_input_noise_cfg, pac_cfg.seed + 200000 + k)
            iter_time = time.time() - iter_start
            total_time = time.time() - total_start
            history.append({
                "iter": k,
                "p_lb": p_lb,
                "u_local": cert_res.u_local,
                "c_bc": cert_res.c_bc,
                "epsilon": eps,
                "n_pre": cert_res.n_pre,
                "n_post": cert_res.n_post,
                "n_trans": cert_res.n_trans,
                "loss_nonneg": cert_res.loss_nonneg,
                "loss_pre": cert_res.loss_pre,
                "loss_post": cert_res.loss_post,
                "loss_dyn": cert_res.loss_dyn,
                "loss_obj": cert_res.loss_obj,
                "epochs_used": cert_res.epochs_used,
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
            print("[iter {:04d}] p_lb={:.3f} u={:.3f} c={:.4f} "
                  "n_pre={} n_post={} n_trans={} cert_safe={:.3f} cert_goal={:.3f} cert_fail={:.3f} "
                  "safe_eval={:.3f} goal_eval={:.3f} fail_eval={:.3f} avg_ret={:.2f} "
                  "iter_time={:.2f}s total_time={:.2f}s "
                  "ln={:.2e} lp={:.2e} lq={:.2e} ld={:.2e} lo={:.2e} max_vio={:.2e} ep={:d} status={}".format(
                      k, p_lb, cert_res.u_local, cert_res.c_bc,
                      cert_res.n_pre, cert_res.n_post, cert_res.n_trans,
                      cert_safe_rate, cert_goal_rate, cert_fail_rate,
                      eval_stats["safe_rate"], eval_stats["goal_rate"], eval_stats["fail_rate"],
                      eval_stats["avg_return"], iter_time, total_time,
                      cert_res.loss_nonneg, cert_res.loss_pre, cert_res.loss_post,
                      cert_res.loss_dyn, cert_res.loss_obj,
                      cert_res.max_violation, cert_res.epochs_used, cert_res.status))

    if final_result is None:
        total_time = time.time() - total_start
        eval_stats = evaluate_policy(env_eval, agent, 200, eval_input_noise_cfg, pac_cfg.seed + 300000)
        final_result = {
            "outer_iter": pac_cfg.max_outer_iters - 1,
            "certificate": last_cert,
            "epsilon": pac_epsilon(pac_cfg.N_cert, pac_cfg.beta),
            "eval": eval_stats,
            "stopped_by_p_lb": False,
            "total_time_sec": total_time,
        }
    return history, final_result


def main():
    set_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_train = PendulumSafetyEnv(
        env_name="Pendulum-v1",
        max_steps=200,
        seed=7,
        safe_angle_deg=90.0,
        unsafe_angle_deg=90.0,
        safe_speed=1.0,
        unsafe_speed=2.5,
        init_angle_deg=10.0,
        init_speed=0.5,
    )
    env_eval = PendulumSafetyEnv(
        env_name="Pendulum-v1",
        max_steps=200,
        seed=123,
        safe_angle_deg=90.0,
        unsafe_angle_deg=90.0,
        safe_speed=1.0,
        unsafe_speed=2.5,
        init_angle_deg=10.0,
        init_speed=0.5,
    )

    ppo_cfg = PPOConfig(
        gamma_rl=0.99,
        gae_lambda=0.95,
        actor_lr=3e-4,
        critic_lr=1e-3,
        batch_size=256,
        clip_ratio=0.20,
        entropy_coef=0.001,
        value_coef=0.50,
        max_grad_norm=0.50,
        gradient_steps_per_iter=50,
        log_std_init=-0.5,
    )

    pac_cfg = PACTrainConfig(
        seed=7,
        max_outer_iters=3000,
        N_cert=100,
        beta=0.05,
        p_min=0.90,
        lambda_bc=50.0,
        gamma_bc=0.01,
        horizon_T=200,
        replay_warmup_episodes=50,
        episodes_per_outer_iter=8,
        test_every=1,
    )

    cert_nn_cfg = CertificateNNConfig(
        hidden_dim=128,
        lr=1e-3,
        epochs=4000,
        batch_size=256,
        lambda_nonneg=10.0,
        lambda_pre=10.0,
        lambda_post=10.0,
        lambda_dyn=10.0,
        lambda_c=1.0,
        tol=1e-3,
        tol_c=1e-4,
    )

    agent = PPOAgent(env_train.state_dim, env_train.action_dim, env_train.max_action, ppo_cfg, device)

    cert_input_noise_cfg = make_gaussian_input_noise(scale=0.02)
    train_input_noise_cfg = make_gaussian_input_noise(scale=0.02)
    eval_input_noise_cfg = make_gaussian_input_noise(scale=0.02)

    history, res = train_pac_guided_ppo(
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

    print("\n==== FINAL RESULT ====")
    print("Stopped by p_lb? {}".format(res["stopped_by_p_lb"]))
    print("Outer iter: {}".format(res["outer_iter"]))
    print("p_lb: {:.4f}".format(res["certificate"].p_lb))
    print("u_local: {:.4f}".format(res["certificate"].u_local))
    print("c_bc: {:.4f}".format(res["certificate"].c_bc))
    print("n_pre: {}".format(res["certificate"].n_pre))
    print("n_post: {}".format(res["certificate"].n_post))
    print("n_trans: {}".format(res["certificate"].n_trans))
    print("loss_nonneg: {:.4e}".format(res["certificate"].loss_nonneg))
    print("loss_pre: {:.4e}".format(res["certificate"].loss_pre))
    print("loss_post: {:.4e}".format(res["certificate"].loss_post))
    print("loss_dyn: {:.4e}".format(res["certificate"].loss_dyn))
    print("loss_obj: {:.4e}".format(res["certificate"].loss_obj))
    print("max_violation: {:.4e}".format(res["certificate"].max_violation))
    print("epochs_used: {}".format(res["certificate"].epochs_used))
    print("epsilon(N,beta): {:.4f}".format(res["epsilon"]))
    print("safe_eval: {:.4f}".format(res["eval"]["safe_rate"]))
    print("goal_eval: {:.4f}".format(res["eval"]["goal_rate"]))
    print("fail_eval: {:.4f}".format(res["eval"]["fail_rate"]))
    print("avg_return: {:.4f}".format(res["eval"]["avg_return"]))
    print("total_time_sec: {:.2f}".format(res["total_time_sec"]))

    print("\n==== ROBUSTNESS SWEEP EXAMPLE ====")
    uniform_results = robustness_sweep(env_eval, agent, "uniform",
                                       scales=[0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20],
                                       n_eval=50)
    gaussian_results = robustness_sweep(env_eval, agent, "gaussian",
                                        scales=[0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.20],
                                        n_eval=50)
    print("Uniform noise:", uniform_results)
    print("Gaussian noise:", gaussian_results)


if __name__ == "__main__":
    main()
