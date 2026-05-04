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
    def __init__(self, state_dim, action_dim, max_action, hidden=256):
        super().__init__()
        self.max_action = float(max_action)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.max_action * self.net(x)


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
class DDPGConfig:
    gamma_rl: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    batch_size: int = 128
    exploration_std: float = 0.15
    exploration_clip: float = 0.5
    gradient_steps_per_iter: int = 50
    replay_capacity: int = 200000


class DDPGAgent(object):
    def __init__(self, state_dim, action_dim, max_action, cfg, device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.max_action = float(max_action)

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_targ = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_targ.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_targ = Critic(state_dim, action_dim).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.replay = ReplayBuffer(cfg.replay_capacity, state_dim, action_dim)

    def select_action(self, state, deterministic=False, input_noise_cfg=None, rng=None):
        noisy_state = apply_input_noise(state, input_noise_cfg, rng)
        s = torch.tensor(noisy_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            if rng is None:
                rng = np.random.default_rng()
            noise = rng.normal(0.0, self.cfg.exploration_std, size=action.shape)
            noise = np.clip(noise, -self.cfg.exploration_clip, self.cfg.exploration_clip)
            action = action + noise
        action = np.clip(action, -self.max_action, self.max_action).astype(np.float32)
        return action, action.copy(), noisy_state

    def update(self):
        if self.replay.size < self.cfg.batch_size:
            return
        batch = self.replay.sample(self.cfg.batch_size, self.device)
        with torch.no_grad():
            next_a = self.actor_targ(batch["next_state"])
            target_q = self.critic_targ(batch["next_state"], next_a)
            y = batch["reward"] + self.cfg.gamma_rl * (1.0 - batch["done"]) * target_q

        q = self.critic(batch["state"], batch["action"])
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pred_a = self.actor(batch["state"])
        actor_loss = -self.critic(batch["state"], pred_a).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.actor_targ)
        self.soft_update(self.critic, self.critic_targ)

    def soft_update(self, src, tgt):
        tau = self.cfg.tau
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)


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


def standard_error(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


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
        "safe_sem": standard_error(safe),
        "goal_sem": standard_error(goal),
        "fail_sem": standard_error(fail),
        "avg_return": float(np.mean(returns)),
        "return_sem": standard_error(returns),
        "returns": [float(x) for x in returns],
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
            "safe_sem": stats["safe_sem"],
            "goal_sem": stats["goal_sem"],
            "fail_sem": stats["fail_sem"],
            "avg_return": stats["avg_return"],
            "return_sem": stats["return_sem"],
            "returns": stats["returns"],
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


def train_pac_guided_ddpg(env_train, env_eval, agent, pac_cfg, cert_nn_cfg, device,
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

        terminal_bonus = -pac_cfg.lambda_bc * cert_res.c_bc
        for _ in range(pac_cfg.episodes_per_outer_iter):
            rollout_episode(env_train, agent, False, terminal_bonus, True, train_input_noise_cfg, rng)
            for _ in range(agent.cfg.gradient_steps_per_iter):
                agent.update()

        if (k % pac_cfg.test_every) == 0 or (k == pac_cfg.max_outer_iters - 1):
            eval_stats = evaluate_policy(env_eval, agent, 100, eval_input_noise_cfg, pac_cfg.seed + 200000 + k)
            iter_time = time.time() - iter_start
            total_time = time.time() - total_start

            if eval_stats["safe_rate"] == 1.0:
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
                "return_sem": eval_stats["return_sem"],
                "returns": eval_stats["returns"],
                "iter_time": iter_time,
                "total_time": total_time,
            })
            print("[iter {:04d}] p_lb={:.3f} u={:.3f} c={:.4f} "
                  "n_pre={} n_post={} n_trans={} cert_safe={:.3f} cert_goal={:.3f} cert_fail={:.3f} "
                  "safe_eval={:.3f} goal_eval={:.3f} fail_eval={:.3f} avg_ret={:.2f} sem_ret={:.2f} "
                  "iter_time={:.2f}s total_time={:.2f}s "
                  "ln={:.2e} lp={:.2e} lq={:.2e} ld={:.2e} lo={:.2e} max_vio={:.2e} ep={:d} status={}".format(
                      k, p_lb, cert_res.u_local, cert_res.c_bc,
                      cert_res.n_pre, cert_res.n_post, cert_res.n_trans,
                      cert_safe_rate, cert_goal_rate, cert_fail_rate,
                      eval_stats["safe_rate"], eval_stats["goal_rate"], eval_stats["fail_rate"],
                      eval_stats["avg_return"], eval_stats["return_sem"], iter_time, total_time,
                      cert_res.loss_nonneg, cert_res.loss_pre, cert_res.loss_post,
                      cert_res.loss_dyn, cert_res.loss_obj,
                      cert_res.max_violation, cert_res.epochs_used, cert_res.status))
            print("[iter {:04d} returns] returns={}".format(k, eval_stats["returns"]))

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

    ddpg_cfg = DDPGConfig(
        gamma_rl=0.99,
        tau=0.005,
        actor_lr=1e-3,
        critic_lr=1e-3,
        batch_size=128,
        exploration_std=0.20,
        exploration_clip=0.5,
        gradient_steps_per_iter=50,
        replay_capacity=200000,
    )

    pac_cfg = PACTrainConfig(
        seed=7,
        max_outer_iters=3000,
        N_cert=100,
        beta=0.05,
        p_min=0.90,
        lambda_bc=0,
        gamma_bc=0.01,
        horizon_T=200,
        replay_warmup_episodes=50,
        episodes_per_outer_iter=8,
        test_every=1,
    )

    cert_nn_cfg = CertificateNNConfig(
        hidden_dim=128,
        lr=1e-3,
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

    agent = DDPGAgent(env_train.state_dim, env_train.action_dim, env_train.max_action, ddpg_cfg, device)

    cert_input_noise_cfg = make_gaussian_input_noise(scale=0.02)
    train_input_noise_cfg = make_gaussian_input_noise(scale=0.02)
    eval_input_noise_cfg = make_gaussian_input_noise(scale=0.02)

    history, res = train_pac_guided_ddpg(
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
    print("return_sem: {:.4f}".format(res["eval"]["return_sem"]))
    print("returns:", res["eval"]["returns"])
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
