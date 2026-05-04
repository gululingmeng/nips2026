import re
import ast
import matplotlib.pyplot as plt

def extract_robustness_data(file_path):
    """
    - Uniform noise: [...]
    - Gaussian noise: [...]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    data = {}

    uniform_match = re.search(r"Uniform noise:\s*(\[[\s\S]*?\])", text)
    gaussian_match = re.search(r"Gaussian noise:\s*(\[[\s\S]*?\])", text)

    data["uniform"] = ast.literal_eval(uniform_match.group(1)) if uniform_match else []
    data["gaussian"] = ast.literal_eval(gaussian_match.group(1)) if gaussian_match else []

    return data


def get_xy(noise_list, y_key="goal_rate"):
    x = [item["scale"] for item in noise_list]
    y = [item[y_key] for item in noise_list]
    return x, y


file_ddpg = "NN_switch_0.005"
file_guided = "guided_switch_0.005"

ddpg_data = extract_robustness_data(file_ddpg)
guided_data = extract_robustness_data(file_guided)

x1, y1 = get_xy(ddpg_data["uniform"], "goal_rate")
x2, y2 = get_xy(guided_data["uniform"], "goal_rate")

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, marker='o', linewidth=2, label='DDPG')
plt.plot(x2, y2, marker='s', linewidth=2, label='PAC-guided')
plt.xlabel("scale")
plt.ylabel("goal_rate")
plt.title("Uniform Noise: scale vs goal_rate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

x1, y1 = get_xy(ddpg_data["gaussian"], "goal_rate")
x2, y2 = get_xy(guided_data["gaussian"], "goal_rate")

plt.figure(figsize=(8, 5))
plt.plot(x1, y1, marker='o', linewidth=2, label='DDPG')
plt.plot(x2, y2, marker='s', linewidth=2, label='PAC-guided')
plt.xlabel("scale")
plt.ylabel("goal_rate")
plt.title("Gaussian Noise: scale vs goal_rate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
