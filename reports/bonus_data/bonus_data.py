import re

import pandas as pd
from matplotlib import pyplot as plt, cycler

# Create CSV
NUM_TESTS = 6
DATA_FOLDER = 'data/'
CSVS_FOLDER = 'csvs/'
PLOTS_FOLDER = 'plots/'
algorithms = ['A-Star.txt', 'BFS.txt', 'DFS.txt', 'Greedy.txt']
board_sizes = ['4', '5', '6', '7', '8', '9', '10']
heuristics = ['1.0', '1.2', '1.4', '1.6', '1.8', '2.0']
csvs = {}

for board_size in board_sizes:
    for algo in algorithms:
        if board_size in ['4', '5', '6']:
            algo_str = algo.lower().replace('.txt', '')
            result_csv = CSVS_FOLDER + board_size + '/' + algo_str + '_result.csv'
            csvs[board_size + '_' + algo_str] = result_csv
            rf = open(result_csv, 'w+')
            rf.write('input,time,memory,generated,expanded\n')
            with open(DATA_FOLDER + board_size + '/' + algo) as f:
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
        else:
            if algo == 'DFS.txt' or algo == 'BFS.txt':
                continue
            for h in heuristics:
                algo_str = algo.lower().replace('.txt', '')
                result_csv = CSVS_FOLDER + board_size + '/' + h + '_' + algo_str + '_result.csv'
                csvs[board_size + '_' + algo_str + '_' + h] = result_csv
                rf = open(result_csv, 'w+')
                rf.write('input,time,memory,generated,expanded\n')
                with open(DATA_FOLDER + board_size + '/' + h + '/' + algo) as f:
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
custom_cycler = (cycler(color=['r', 'b', 'g', 'k', 'y', 'm']) +
                 cycler(linestyle=['-', '--', ':', '-.', '-', '--']))
labels = {'time': 'Execution Time(s)', 'memory': 'Used Memory (kB)', 'generated': 'Generated Nodes', 'expanded': 'Expanded Nodes'}

for algo, csv in csvs.items():
    df = pd.read_csv(csv, index_col=False)
    algo_df[algo] = df

average_dict = {}
for algo in algorithms:
    if algo == 'BFS.txt' or algo == 'DFS.txt':
        continue
    fig, ax = plt.subplots()
    ax.set_prop_cycle(custom_cycler)
    algo_str = algo.lower().replace('.txt', '')
    for h in heuristics:
        aux = '10_' + algo_str + '_' + h
        time = algo_df[aux]['time']
        ax.plot([i for i in range(NUM_TESTS)], time, label=h, alpha=0.5)
    ax.set_xticks([i for i in range(1, NUM_TESTS + 1)])
    ax.xaxis.grid(alpha=0.3)
    ax.legend(loc='best')
    ax.set_xlabel('Input')
    ax.set_ylabel(labels['time'])
    plt.savefig(f'{PLOTS_FOLDER}{algo_str}_h_plot.pdf')
    plt.cla()
