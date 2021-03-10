import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="scattered-ccur-0.75-lstm-3-alr-3e-05-clr-3e-05-03072021_090203")

args = parser.parse_args()

run_names = ["scattered-ccur-0.75-lstm-3-alr-3e-05-clr-3e-05-03072021_090203",
"scattered-ccur-0.75-lstm-5-alr-3e-05-clr-3e-05-03082021_182550"]
runs_data = []

for run_name in run_names:
    with open("./"+run_name+"-run.json") as f:
        run_data = json.load(f)
        runs_data.append(run_data)

t= range(len(runs_data[0]))

fig, axs = plt.subplots(3, 2)
axs[0][0].plot(t, list(map(lambda x: x["reward"], runs_data[0])), 
    t, list(map(lambda x: x["reward"], runs_data[1])))
axs[0][0].set_xlabel('step')
axs[0][0].set_ylabel('reward acquired')
axs[0][0].grid(True)

runs_cumulated_rewards = []

for i in range(len(runs_data)):
    reward_sum = 0
    cumulated_rewards = []
    for j in range(len(run_data)):
        reward_sum+= runs_data[i][j]["reward"]
        cumulated_rewards.append(reward_sum) 
    runs_cumulated_rewards.append(cumulated_rewards)

axs[0][1].plot(t, runs_cumulated_rewards[0], t , runs_cumulated_rewards[1])
axs[0][1].set_xlabel('step')
axs[0][1].set_ylabel('cumulated reward')
axs[0][1].grid(True)

axs[1][0].plot(t, list(map(lambda x: x["input_set_size"], runs_data[0])),
    t, list(map(lambda x: x["input_set_size"], runs_data[1])))
axs[1][0].set_xlabel('step')
axs[1][0].set_ylabel('input set size')
axs[1][0].set_yscale("log")
axs[1][0].grid(True)

axs[1][1].plot(t, list(map(lambda x: x["output_set_count"], runs_data[0])),
    t, list(map(lambda x: x["output_set_count"], runs_data[1])))
axs[1][1].set_xlabel('step')
axs[1][1].set_ylabel('output set count')
axs[1][1].grid(True)

axs[2][0].plot(t, list(map(lambda x: x["output_set_average_size"], runs_data[0])),
    t, list(map(lambda x: x["output_set_average_size"], runs_data[1])))
axs[2][0].set_xlabel('step')
axs[2][0].set_ylabel('output set average size')
axs[2][0].set_yscale("log")
axs[2][0].grid(True)

operators = ["by_facet", "by_superset", "by_neighbors", "by_distribution"]
axs[2][1].plot(t, list(map(lambda x: x["operator"], runs_data[0])),
    t, list(map(lambda x: x["operator"], runs_data[1])))
axs[2][1].set_xlabel('step')
axs[2][1].set_ylabel('Operators')
axs[2][1].grid(True)

fig.tight_layout()
plt.show()