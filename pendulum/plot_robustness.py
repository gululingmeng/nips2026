import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LINE_WIDTH = 4
MARKER_SIZE = 9
LABEL_SIZE = 18
TITLE_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16


def standard_error(values):
    values = [float(v) for v in values]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def extract_literal_list_after(text, marker):
    start = text.find(marker)
    if start < 0:
        return []

    start = text.find("[", start + len(marker))
    if start < 0:
        return []

    depth = 0
    quote = None
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if quote is not None:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = None
            continue

        if ch in ("'", '"'):
            quote = ch
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return ast.literal_eval(text[start:idx + 1])

    raise ValueError("Unclosed list after marker: {}".format(marker))


def extract_robustness_data(file_path):
    """
    - Uniform noise: [...]
    - Gaussian noise: [...]

    {
        "uniform": [{"scale": ..., "safe_rate": ...}, ...],
        "gaussian": [{"scale": ..., "safe_rate": ...}, ...]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "uniform": extract_literal_list_after(text, "Uniform noise:"),
        "gaussian": extract_literal_list_after(text, "Gaussian noise:"),
    }


def get_xy_sem(noise_list, y_key="safe_rate", sem_key="safe_sem"):
    x = [item["scale"] for item in noise_list]
    y = [item[y_key] for item in noise_list]
    sem = []
    for item in noise_list:
        if sem_key in item:
            sem.append(float(item[sem_key]))
        elif y_key == "avg_return" and "returns" in item:
            sem.append(standard_error(item["returns"]))
        else:
            sem.append(None)
    return x, y, sem


def plot_mean_sem(x, y, sem, marker, label):
    line, = plt.plot(x, y, marker=marker, linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=label)
    sem_arr = np.asarray([np.nan if s is None else s for s in sem], dtype=np.float64)
    if np.isfinite(sem_arr).any():
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        plt.fill_between(
            x_arr,
            y_arr - sem_arr,
            y_arr + sem_arr,
            color=line.get_color(),
            alpha=0.18,
            linewidth=0,
        )
    return line


file_ddpg = SCRIPT_DIR / "DDPG"
file_guided = SCRIPT_DIR / "DDPG-guided"

ddpg_data = extract_robustness_data(file_ddpg)
guided_data = extract_robustness_data(file_guided)

x1, y1, sem1 = get_xy_sem(ddpg_data["uniform"], "safe_rate", "safe_sem")
x2, y2, sem2 = get_xy_sem(guided_data["uniform"], "safe_rate", "safe_sem")

plt.figure(figsize=(8, 5))
plot_mean_sem(x1, y1, sem1, marker='o', label='DDPG')
plot_mean_sem(x2, y2, sem2, marker='s', label='PAC-guided')
plt.xlabel("scale", fontsize=LABEL_SIZE)
plt.ylabel("safe_rate", fontsize=LABEL_SIZE)
plt.title("Uniform Noise: safe_rate (mean +/- SEM)", fontsize=TITLE_SIZE)
plt.tick_params(axis='both', labelsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "uniform.png", dpi=300, bbox_inches="tight")
plt.show()

x1, y1, sem1 = get_xy_sem(ddpg_data["gaussian"], "safe_rate", "safe_sem")
x2, y2, sem2 = get_xy_sem(guided_data["gaussian"], "safe_rate", "safe_sem")

plt.figure(figsize=(8, 5))
plot_mean_sem(x1, y1, sem1, marker='o', label='DDPG')
plot_mean_sem(x2, y2, sem2, marker='s', label='PAC-guided')
plt.xlabel("scale", fontsize=LABEL_SIZE)
plt.ylabel("safe_rate", fontsize=LABEL_SIZE)
plt.title("Gaussian Noise: safe_rate (mean +/- SEM)", fontsize=TITLE_SIZE)
plt.tick_params(axis='both', labelsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "gauss.png", dpi=300, bbox_inches="tight")
plt.show()
