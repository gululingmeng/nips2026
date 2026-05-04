# Certificate-Guided PAC Policy Learning under Temporal Logic Specifications

This repository contains the experiment code and logs for:

**Certificate-Guided PAC Policy Learning under Temporal Logic Specifications**

## Abstract

Learning neural control policies for stochastic neural network-controlled systems remains challenging when the desired behavior is specified by temporal logic and the system dynamics and disturbance distribution are only accessible through sampled trajectories. This paper proposes a certificate-guided PAC policy-learning framework for stochastic neural network-controlled systems under safe LTL$_f$ specifications. We first decompose the violation of a safe LTL$_f$ formula through the accepting runs of the DFA for its negation, and construct neural barrier certificates for the resulting sequential reachability conditions. Based on sampled trajectories, we derive a computable lower bound on the specification satisfaction probability and validate it with a Probably Approximately Correct (PAC) guarantee under unknown dynamics and perturbations. We then integrate this certificate module into policy learning as a plug-and-play training--verification loop for generic deep reinforcement learning algorithms. Experiments on multiple control benchmarks show that the proposed PAC-guided variants improve robustness under unseen disturbances, and provide a valid and tight lower-bound guarantee for probability.

## Repository Layout

```text
.
|-- 10-room/      Ten-room benchmark scripts and experiment logs.
|-- cartpole/     CartPole benchmark scripts, logs, and generated figures.
|-- pendulum/     Pendulum benchmark scripts, logs, and generated figures.
|-- truck/        Truck benchmark scripts and experiment logs.
`-- requirement.txt
```

Each benchmark directory contains self-contained Python scripts for baseline and certificate-guided training. The extensionless files such as `DDPG`, `DDPG-guided`, `DDPG-guarantee`, `TD3`, `TD3-guided`, `SAC`, `SAC-guided`, `PPO`, and `PPO-guided` are saved console logs used by the plotting scripts.

## Benchmarks and Scripts

| Directory | Main contents |
| --- | --- |
| `cartpole/` | CartPole DDPG, TD3, SAC, PPO, certificate-guided variants, DDPG guarantee variant, performance and robustness plots |
| `pendulum/` | Pendulum DDPG, TD3, SAC, PPO, certificate-guided variants, DDPG guarantee variant, performance and robustness plots |
| `truck/` | Truck DDPG, TD3, SAC, PPO, certificate-guided variants, and DDPG guarantee variant |
| `10-room/` | Ten-room DDPG, TD3, SAC, PPO, certificate-guided variants, and DDPG guarantee variant |

Current script naming follows this pattern:

```text
cartpole/cartpole_DDPG.py
cartpole/cartpole_DDPG_guarantee.py
cartpole/cartpole_DDPG_guided.py
cartpole/cartpole_DDPG_guided_100.py
cartpole/cartpole_DDPG_guided_150.py
cartpole/cartpole_TD3.py
cartpole/cartpole_TD3_guided.py
cartpole/cartpole_SAC.py
cartpole/cartpole_SAC_guided.py
cartpole/cartpole_ppo.py
cartpole/cartpole_ppo_guided.py

pendulum/pendulum_DDPG.py
pendulum/pendulum_DDPG_guarantee.py
pendulum/pendulum_DDPG_guided.py
pendulum/pendulum_DDPG_guided_50.py
pendulum/pendulum_DDPG_guided_150.py
pendulum/pendulum_TD3.py
pendulum/pendulum_TD3_guided.py
pendulum/pendulum_SAC.py
pendulum/pendulum_SAC_guided.py
pendulum/pendulum_ppo.py
pendulum/pendulum_ppo_guided.py

truck/truck_DDPG.py
truck/truck_DDPG_guarantee.py
truck/truck_DDPG_guided.py
truck/truck_DDPG_guided_50.py
truck/truck_DDPG_guided_100.py
truck/truck_TD3.py
truck/truck_TD3_guided.py
truck/truck_SAC.py
truck/truck_SAC_guided.py
truck/truck_ppo.py
truck/truck_ppo_guided.py

10-room/10room_DDPG.py
10-room/10room_DDPG_guarantee.py
10-room/10room_DDPG_guided.py
10-room/10room_DDPG_guided_50.py
10-room/10room_DDPG_guided_150.py
10-room/10room_TD3.py
10-room/10room_TD3_guided.py
10-room/10room_SAC.py
10-room/10room_SAC_guided.py
10-room/10room_ppo.py
10-room/10room_ppo_guided.py
```

Most scripts have a `main()` block with experiment hyperparameters set directly in the file. They print training progress, a `FINAL RESULT` summary, and a robustness sweep under uniform and Gaussian disturbances.

## Environment

The dependency file was exported from the local conda environment named `Certificate-Guided`.

```bash
conda create -n Certificate-Guided python=3.7.12
conda activate Certificate-Guided
pip install -r requirement.txt
```

Important packages include PyTorch, Gym/Gymnasium, NumPy, SciPy, Matplotlib, CVXPY, TensorFlow, Stable-Baselines3, and Gurobi Python bindings. CUDA is optional; the scripts use GPU automatically when `torch.cuda.is_available()` is true.

## Running Experiments

Run scripts from the repository root or from the corresponding benchmark directory.

```bash
conda activate Certificate-Guided

python cartpole/cartpole_DDPG_guided.py
python pendulum/pendulum_DDPG_guided.py
python truck/truck_DDPG_guided.py
python 10-room/10room_DDPG_guided.py
```

Common baseline and guided variants can be run as follows:

```bash
python cartpole/cartpole_DDPG.py
python cartpole/cartpole_DDPG_guided.py
python cartpole/cartpole_DDPG_guarantee.py
python cartpole/cartpole_TD3.py
python cartpole/cartpole_TD3_guided.py
python cartpole/cartpole_SAC.py
python cartpole/cartpole_SAC_guided.py
python cartpole/cartpole_ppo.py
python cartpole/cartpole_ppo_guided.py
```

The main experiment settings, including seeds, horizons, PAC sample sizes, disturbance scales, certificate-training epochs, and stopping thresholds, are configured inside each script's `main()` function.

## Outputs

A typical run prints:

- per-iteration evaluation statistics such as `safe_rate`, `goal_rate`, `phi_rate`, `avg_return`, and standard error when available;
- certificate quantities such as `p_lb`, `u_local`, `c_bc`, losses, maximum violation, and epochs used;
- final PAC-related values such as `epsilon(N,beta)`;
- robustness sweeps printed as `Uniform noise: [...]` and `Gaussian noise: [...]`.

To save a run log for plotting, redirect stdout to one of the log files expected by the plotting scripts:

```bash
python cartpole/cartpole_DDPG.py > cartpole/DDPG
python cartpole/cartpole_DDPG_guided.py > cartpole/DDPG-guided
```

## Plotting

CartPole and Pendulum include scripts for performance and robustness figures:

```bash
python cartpole/plot_performance.py
python cartpole/plot_robustness.py

python pendulum/plot_performance.py
python pendulum/plot_robustness.py
```

These scripts read the saved log files in the same directory and write figures such as `performance.png`, `uniform.png`, and `gauss.png`.

The current `truck/` directory contains experiment scripts and log files, but no `truck/plot_robustness.py` plotting script. To plot Truck robustness results, add or restore a plotting script that reads `truck/DDPG` and `truck/DDPG-guided`, following the CartPole/Pendulum plotting format.

## Reproducibility Notes

- The scripts set fixed random seeds in `main()`, but GPU kernels and some environment behavior may still introduce small numerical variation.
- Long-running scripts may take substantial time because policy learning and neural certificate training are executed in the same loop.
- PAC-guided variants are designed as training--verification loops: the policy learner collects data, the certificate module estimates a lower bound on satisfaction probability, and training can stop once the configured lower-bound target is met.
