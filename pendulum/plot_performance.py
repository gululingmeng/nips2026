import ast
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
LINE_WIDTH = 4
LABEL_SIZE = 18
TITLE_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 16
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def standard_error(values):
    values = [float(v) for v in values]
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def extract_iter_avg_ret_with_final(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    by_iter = {}

    iter_pattern = re.compile(
        rf"\[iter\s+(\d+)\].*?avg_ret=({FLOAT_RE})(?:\s+sem_ret=({FLOAT_RE}))?"
    )
    for match in iter_pattern.finditer(text):
        it = int(match.group(1))
        by_iter.setdefault(it, {})["avg"] = float(match.group(2))
        if match.group(3) is not None:
            by_iter[it]["sem"] = float(match.group(3))

    returns_pattern = re.compile(r"\[iter\s+(\d+)\s+returns\]\s+returns=(\[[^\n]*\])")
    for match in returns_pattern.finditer(text):
        it = int(match.group(1))
        returns = ast.literal_eval(match.group(2))
        entry = by_iter.setdefault(it, {})
        entry.setdefault("avg", float(np.mean(returns)))
        entry["sem"] = standard_error(returns)

    outer_iter_match = re.search(r'Outer iter:\s*(\d+)', text)
    final_avg_match = re.search(rf'avg_return:\s*({FLOAT_RE})', text)
    final_sem_match = re.search(rf'return_sem:\s*({FLOAT_RE})', text)
    final_returns_match = re.search(r"^returns:\s*(\[[^\n]*\])", text, re.MULTILINE)

    if outer_iter_match and final_avg_match:
        final_iter = int(outer_iter_match.group(1))
        final_avg = float(final_avg_match.group(1))
        entry = by_iter.setdefault(final_iter, {})
        entry["avg"] = final_avg
        if final_sem_match:
            entry["sem"] = float(final_sem_match.group(1))
        elif final_returns_match:
            entry["sem"] = standard_error(ast.literal_eval(final_returns_match.group(1)))

    points = [(it, item["avg"], item.get("sem")) for it, item in by_iter.items() if "avg" in item]
    points.sort(key=lambda x: x[0])

    iters = [p[0] for p in points]
    avg_rets = [p[1] for p in points]
    sem_rets = [p[2] for p in points]
    return iters, avg_rets, sem_rets


def plot_mean_sem(x, y, sem, label, color=None):
    line, = plt.plot(x, y, linewidth=LINE_WIDTH, label=label, color=color)
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


file1 = SCRIPT_DIR / "NN_DDPG"
file2 = SCRIPT_DIR / "guided_DDPG"

iters1, avg_rets1, sem_rets1 = extract_iter_avg_ret_with_final(file1)
iters2, avg_rets2, sem_rets2 = extract_iter_avg_ret_with_final(file2)

plt.figure(figsize=(10, 6))
plot_mean_sem(iters1, avg_rets1, sem_rets1, label='DDPG')
plot_mean_sem(iters2, avg_rets2, sem_rets2, label='PAC-guided')

plt.xlabel("iter", fontsize=LABEL_SIZE)
plt.ylabel("avg_return", fontsize=LABEL_SIZE)
plt.title("iter vs avg_return (mean +/- SEM)", fontsize=TITLE_SIZE)
plt.tick_params(axis='both', labelsize=TICK_SIZE)
plt.legend(fontsize=LEGEND_SIZE)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "performance.png", dpi=300, bbox_inches="tight")
plt.show()
