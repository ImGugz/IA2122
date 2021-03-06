import sys
import random
import numpy as np
import math
import re
import os
import copy

MIN_DELETE_FRACTION = 5/7
MAX_DELETE_FRACTION = 6/7

def to_string(table):
    res = ""
    for i in range(len(table)):
        for j in range(len(table)):
            res += str(table[i][j]) + "\t"
        res = res[:-1] + "\n"
    return res

file = sys.argv[1]
board = []

with open (file, "r") as f:
    while f.readline() != "Solution:\n":
        pass
    for line in f.readlines():
        split = re.split("\s{1,}", line)
        if split[-1] == '':
            split = split[:-1]
        if split[0] == '':
            split = split[1:]
        board.append([int(n) for n in split])

n = len(board)
k = random.randint(math.ceil(n * n * MIN_DELETE_FRACTION), math.floor(n * n * MAX_DELETE_FRACTION))

old_board = copy.deepcopy(board)

# randomly delete board entries
while k > 0:
    to_delete = random.randint(0, n * n - 1)
    i, j = to_delete % n, to_delete // n
    if board[i][j] != 0:
        board[i][j] = 0
        k -= 1

i = 1

while os.path.exists(f"generated/input_{n}_{i}.txt") or os.path.exists(f"generated/output_{n}_{i}.txt") :
    i += 1

with open(f"generated/input_{n}_{i}.txt", "w") as f:
    f.write(f"{n}\n")
    f.write(to_string(board))
    f.close()

with open(f"generated/output_{n}_{i}.txt", "w") as f:
    f.write(to_string(old_board))
    f.close()

print(f"Created new files generated/input_{n}_{i}.txt and generated/output_{n}_{i}.txt")