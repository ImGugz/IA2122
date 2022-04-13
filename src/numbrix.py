# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 20:
# 95565 Duarte Almeida
# 95587 Gustavo Aguiar
import copy
import sys
import time

from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search

class NumbrixState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    board_size = 0
    board = []
    numbers_to_go = set()
    free_adjacents = set()
    vertical_adjacents = 0
    horizontal_adjacents = 0
    # board_size, board, initial_numbers
    def __init__(self, *args):
        self.board = args[0]
        self.board_size = args[1]
        if len(args) == 3:
            self.numbers_to_go = {i for i in range(1, self.board_size * self.board_size + 1)} - args[2]
            self.free_adjacents = set([(i, j) for j in range(self.board_size) for i in range(self.board_size)
                                       if self.has_adjacents(i, j)])
            for i in range(self.board_size):
                self.horizontal_adjacents += sum(
                    [1 if ((self.board[i][j] == self.board[i][j + 1] + 1 or self.board[i][j] == self.board[i][j + 1] - 1) and self.board[i][j] != 1) else 0 for j in
                     range(self.board_size - 1)])
                self.vertical_adjacents += sum(
                    [1 if ((self.board[j][i] == self.board[j + 1][i] + 1 or self.board[j][i] == self.board[j + 1][i] - 1) and self.board[i][j] != 1) else 0 for j in
                     range(self.board_size - 1)])
        else:
            self.numbers_to_go = args[2]
            self.free_adjacents = args[3]
            self.vertical_adjacents = args[4]
            self.horizontal_adjacents = args[5]

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        try:
            return self.board[row][col]
        except IndexError:
            return None

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente abaixo e acima,
        respectivamente. """
        results = []
        for dir in [1, -1]:
            try:
                results.append(self.board[row + dir][col])
            except IndexError:
                results.append(None)
        return tuple(results)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """
        results = []
        for dir in [-1, 1]:
            try:
                results.append(self.board[row][col + dir])
            except IndexError:
                results.append(None)
        return tuple(results)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """
        file_board = []
        try:
            with open(filename) as f:
                initial_numbers = set()
                board_size = int(f.readline())
                for line in f.readlines():
                    split = line.split('\t')
                    [initial_numbers.add(int(i)) for i in split if int(i) != 0]
                    file_board.append([int(i) for i in split])
            return Board(file_board, board_size, initial_numbers)
        except IOError:  # Couldn't open input file
            print("Something went wrong while attempting to read file.")
            sys.exit(-1)

    # TODO: outros metodos da classe

    def to_string(self):
        return '\n'.join('\t'.join(f'{i}' for i in x) for x in self.board) + '\n'

    def get_adjacents(self, row, col):
        return list(filter(lambda x: 0 <= x[0] < self.board_size and 0 <= x[1] < self.board_size
                           , [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]))

    def has_adjacents(self, row, col):
        return (0 in self.adjacent_vertical_numbers(row, col) or 0
            in self.adjacent_horizontal_numbers(row, col)) and self.get_number(row, col) != 0

    def apply_action(self, action):
        (row, col, value) = action
        new_board = copy.deepcopy(self)
        new_board.board[row][col] = value
        new_board.numbers_to_go = new_board.numbers_to_go - {value}
        new_board.free_adjacents = new_board.free_adjacents - set(new_board.get_adjacents(row, col))
        new_board.free_adjacents = new_board.free_adjacents.union(set(el for el in new_board.get_adjacents(row, col) + [(row, col)]
                                   if new_board.has_adjacents(el[0], el[1])))
        new_board.vertical_adjacents = new_board.vertical_adjacents + sum([1 if j == col and abs(new_board.board[row][col] - new_board.board[i][j]) == 1 else 0
                                        for i, j in new_board.get_adjacents(row, col)])
        new_board.horizontal_adjacents = new_board.horizontal_adjacents + sum([1 if i == row and abs(new_board.board[row][col] - new_board.board[i][j]) == 1 else 0
                                          for i, j in new_board.get_adjacents(row, col)])
        return NumbrixState(new_board)

class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        actions = []
        for (row, col) in state.board.free_adjacents:
            adj_value = state.board.get_number(row, col)
            adjacents = state.board.get_adjacents(row, col)
            for (adj_r, adj_c) in adjacents:
                if state.board.get_number(adj_r, adj_c) != 0:
                    continue
                adj_vector = (adj_r - row + adj_r, adj_c - col + adj_c)
                aux_actions = [-1, 1]
                if (0 <= adj_vector[0] < state.board.board_size) and (0 <= adj_vector[1] < state.board.board_size):
                    adj_vec_num = state.board.get_number(adj_vector[0], adj_vector[1])
                    if adj_vec_num != 0 and adj_vec_num - adj_value != 2:
                        aux_actions.remove(1)
                    if adj_vec_num != 0 and adj_vec_num - adj_value != -2:
                        aux_actions.remove(-1)
                for action in aux_actions:
                    if adj_value + action in state.board.numbers_to_go:
                        actions.append((adj_r, adj_c, adj_value + action))
        return actions

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """
        return state.board.apply_action(action)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. """
        return (state.board.horizontal_adjacents == state.board.board_size * (state.board.board_size - 1)) \
               or (state.board.vertical_adjacents == state.board.board_size * (state.board.board_size - 1))

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        return node.state.board.board_size * (node.state.board.board_size - 1) \
               - max(node.state.board.horizontal_adjacents, node.state.board.vertical_adjacents)

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    tic = time.perf_counter()
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    goal_node = recursive_best_first_search(problem)
    toc = time.perf_counter()

    print(f"RBFS: Programa executado em {toc - tic:0.4f} segundos.")
    print(goal_node.state.board.to_string(), sep="")

