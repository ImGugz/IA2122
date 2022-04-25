import re

import pandas as pd
from matplotlib import pyplot as plt, cycler

# Create CSV
NUM_TESTS = 10
DATA_FOLDER = 'data/'
CSVS_FOLDER = 'csvs/'
PLOTS_FOLDER = 'plots/'
algorithms = ['A-Star.txt', 'BFS.txt', 'DFS.txt', 'Greedy.txt']
csvs = {}
for algo in algorithms:
    algo_str = algo.lower().replace('.txt', '')
    result_csv = CSVS_FOLDER + algo_str + '_result.csv'
    csvs[algo_str] = result_csv
    rf = open(result_csv, 'w+')
    rf.write('input,time,memory,generated,expanded\n')
    with open(DATA_FOLDER + algo) as f:
        i = 0
        while i < NUM_TESTS:
            input_test = f.readline().replace('input', '').rstrip()
            time = re.findall(r'\d+\.\d+', f.readline())[0]
            memory = re.findall(r'\d+', f.readline())[0]
            generated = re.findall(r'\d+', f.readline())[0]
            expanded = re.findall(r'\d+', f.readline())[0]
            if i < NUM_TESTS:
                discard_line = f.readline()
            i += 1
            rf.write(f'{input_test}, {time}, {memory}, {generated}, {expanded}\n')
    rf.close()

# Create DF
algo_df = {}
custom_cycler = (cycler(color=['r', 'b', 'g', 'k']) +
                 cycler(linestyle=['-', '--', ':', '-.']))
labels = {'time': 'Execution Time(s)', 'memory': 'Used Memory (kB)', 'generated': 'Generated Nodes', 'expanded': 'Expanded Nodes'}

for algo, csv in csvs.items():
    df = pd.read_csv(csv, index_col=False)
    algo_df[algo] = df

for col in df:
    if col == 'input':
        continue
    fig, ax = plt.subplots()
    ax.set_prop_cycle(custom_cycler)
    for algo in algo_df.keys():
        ax.plot([i for i in range(NUM_TESTS)], algo_df[algo][col], label=algo, alpha=0.5)
    ax.set_xticks([i for i in range(NUM_TESTS)])
    ax.xaxis.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlabel('Input')
    ax.set_ylabel(labels[col])
    plt.savefig(f'{PLOTS_FOLDER}{col}_plot.pdf')
    plt.cla()
