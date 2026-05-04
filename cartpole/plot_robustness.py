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
        "uniform": [{"scale": ..., "avg_return": ...}, ...],
        "gaussian": [{"scale": ..., "avg_return": ...}, ...]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return {
        "uniform": extract_literal_list_after(text, "Uniform noise:"),
        "gaussian": extract_literal_list_after(text, "Gaussian noise:"),
    }


def get_xy_sem(noise_list, y_key="avg_return", sem_key="return_sem"):
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


file_ddpg = SCRIPT_DIR / "NN_DDPG"
file_pac = SCRIPT_DIR / "guided_DDPG"

ddpg_data = extract_robustness_data(file_ddpg)
pac_data = extract_robustness_data(file_pac)

x1, y1, sem1 = get_xy_sem(ddpg_data["uniform"])
x2, y2, sem2 = get_xy_sem(pac_data["uniform"])

plt.figure(figsize=(8, 5))
plot_mean_sem(x1, y1, sem1, marker='o', label='DDPG')
plot_mean_sem(x2, y2, sem2, marker='s', label='PAC-guided')
plt.xlabel("scale", fontsize=LABEL_SIZE)
plt.ylabel("avg_return", fontsize=LABEL_SIZE)
plt.title("Uniform Noise: avg_return (mean +/- SEM)", fontsize=TITLE_SIZE)
plt.tick_params(axis='both', labelsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "uniform.png", dpi=300, bbox_inches="tight")
plt.show()

x1, y1, sem1 = get_xy_sem(ddpg_data["gaussian"])
x2, y2, sem2 = get_xy_sem(pac_data["gaussian"])

plt.figure(figsize=(8, 5))
plot_mean_sem(x1, y1, sem1, marker='o', label='DDPG')
plot_mean_sem(x2, y2, sem2, marker='s', label='PAC-guided')
plt.xlabel("scale", fontsize=LABEL_SIZE)
plt.ylabel("avg_return", fontsize=LABEL_SIZE)
plt.title("Gaussian Noise: avg_return (mean +/- SEM)", fontsize=TITLE_SIZE)
plt.tick_params(axis='both', labelsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "gauss.png", dpi=300, bbox_inches="tight")
plt.show()
