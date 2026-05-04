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

Each benchmark directory contains self-contained Python scripts for baseline and certificate-guided training. The extensionless files such as `NN_DDPG`, `guided_DDPG`, `guarantee_DDPG`, `pac-guided`, and `pac-guarantee` are saved console logs used by the plotting scripts.

## Benchmarks and Scripts

| Directory | Main contents |
| --- | --- |
| `cartpole/` | CartPole DDPG, TD3, SAC, PPO, PAC-guided variants, guarantee variant, robustness/performance plots |
| `pendulum/` | Pendulum DDPG-style NN, TD3, SAC, PPO, PAC-guided variants, guarantee variant, robustness/performance plots |
| `truck/` | Truck DDPG-style NN, TD3, SAC, PPO, PAC-guided variants, guarantee variant, robustness plot |
| `10-room/` | Ten-room DDPG-style NN, TD3, SAC, PPO, PAC-guided/PAC-guarantee variants |

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

python cartpole/cartpole_NN_guided.py
python pendulum/pendulum_guided.py
python truck/truck_guided.py
python 10-room/10room_pac_guided.py
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
python cartpole/cartpole_NN.py > cartpole/NN_DDPG
python cartpole/cartpole_NN_guided.py > cartpole/guided_DDPG
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

The Truck plotting script currently has hard-coded log names:

```bash
python truck/plot_robustness.py
```

If your logs use names such as `truck/NN_DDPG` and `truck/guided_DDPG`, edit `file_ddpg` and `file_guided` in `truck/plot_robustness.py` before running it.

## Reproducibility Notes

- The scripts set fixed random seeds in `main()`, but GPU kernels and some environment behavior may still introduce small numerical variation.
- Long-running scripts may take substantial time because policy learning and neural certificate training are executed in the same loop.
- PAC-guided variants are designed as training--verification loops: the policy learner collects data, the certificate module estimates a lower bound on satisfaction probability, and training can stop once the configured lower-bound target is met.

