import sys
import random
import numpy as np
import math

MIN_DELETE_FRACTION = 5 / 6
MAX_DELETE_FRACTION = 35 / 36

def concatenate(lst):
    res = ""
    for el in lst:
        res += el
    return res

def horizontal_fill(board, ranges, i, line):
    board[line, :] = ranges[i, :]

def vertical_fill(board, ranges, i, line):
    board[:, line] = ranges[i, :]

def to_string(board):
    res = ""
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            res += str(board[i][j]) + "\t"
        res = res[:-1] + "\n"
    return res

n = int(sys.argv[1])
board = np.array([[0] * n] * n)

ranges = np.array([[j for j in range(i * n + 1, (i + 1) * n + 1)] for i in range(0, n)])
range_indexes = [i for i in range(n)]

# randomly construct a solution instance of numbrix
fill_line = random.choice([horizontal_fill, vertical_fill])
line = 0;

while range_indexes != []:
    index = random.choice(range_indexes)
    range_indexes.remove(index)
    fill_line(board, ranges, index, line)
    line += 1   

k = random.randint(math.ceil(n * n * MIN_DELETE_FRACTION), math.floor(n * n * MAX_DELETE_FRACTION))

# randomly delete board entries 
while k > 0:
    to_delete = random.randint(0, n * n - 1)
    i, j = to_delete % n, to_delete // n
    if board[i][j] != 0:
        board[i][j] = 0
        k -= 1

with open(f"../tests/input_{n}.txt", "w") as f:
    f.write(to_string(board))
    f.close()










    

