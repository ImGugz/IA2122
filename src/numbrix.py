# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

import sys
import copy
import time
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, recursive_best_first_search


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

    board = {}
    board_size = 0
    numbers_to_go = {}
    islands = []
    island_map = {} # Value -> Set #]
    assigned_positions = {}

    def __init__(self, board, board_size):

        self.board_size = board_size
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[(i, j)] = board[i][j]
                self.assigned_positions[board[i][j]] = (i, j)

        self.numbers_to_go = set(i for i in range(1, self.board_size * self.board_size + 1))

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[(i, j)] != 0:
                    self.numbers_to_go.remove(self.board[(i, j)])
                    occupied_neighbors = [pos for pos in self.get_adjacents(i, j) if self.board[pos] != 0 and
                                          (pos[0] < i or (pos[0] == i and pos[1] < j)) and
                                          abs(self.get_number(i, j) - self.get_number(pos[0], pos[1])) == 1]
                    if len(occupied_neighbors) != 0:
                        self.island_map[self.get_number(occupied_neighbors[0][0], occupied_neighbors[0][1])].add(self.board[(i, j)])
                        self.island_map[self.get_number(i, j)] = self.island_map[self.get_number(occupied_neighbors[0][0], occupied_neighbors[0][1])]
                    else:
                        new_set = {self.board[(i, j)]}
                        self.islands.append(new_set)
                        self.island_map[self.get_number(i, j)] = new_set

        self.islands.sort(key = lambda x: min(x))
        #print(self.to_string())
        #print(self.islands)


    def get_adjacents(self, row, col):
        return list(filter(lambda x: 0 <= x[0] < self.board_size and 0 <= x[1] < self.board_size
                           , [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]))

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        try:
            return self.board[row, col]
        except KeyError:
            return None

    def set_number(self, row: int, col: int, value):
        try:
            self.board[row, col] = value
        except KeyError:
            pass

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
                    file_board.append([int(i) for i in split])
            # TODO: Verificar construtor do Board
            return Board(file_board, board_size)
        except IOError:  # Couldn't open input file
            print("Something went wrong while attempting to read file.")
            sys.exit(-1)

    # TODO: outros metodos da classe

    def manhattan_distances(self):
        if len(self.islands) == 1:
            return 0
        else:
            vectors = [(self.assigned_positions[max(self.islands[i])],
                        self.assigned_positions[min(self.islands[i + 1])]) for i in range(len(self.islands) - 1)]
            return max([abs(vector[0][0] - vector[1][0]) + abs(vector[1][0] - vector[1][1])
                         for vector in vectors])

    def to_string(self):
        return '\n'.join('\t'.join(f'{self.get_number(i, j)}' for j in range(self.board_size))
                         for i in range(self.board_size)) + '\n'

    def __copy__(self):
        class Empty(self.__class__):
            def __init__(self): pass
        new_copy = Empty()
        new_copy.__class__ = self.__class__
        return new_copy



class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        self.initial = NumbrixState(board)

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """

        actions = []
        islands_len = len(state.board.islands)

        #print("Initial board")
        #print(state.board.islands)
        #print(state.board.to_string())

        for i in range(islands_len - 1):

            maximum = max(state.board.islands[i])
            maximum_pos = state.board.assigned_positions[maximum]

            minimum = min(state.board.islands[i + 1])
            minimum_pos =state.board.assigned_positions[minimum]

            if maximum + 1 in state.board.numbers_to_go:
                for pos in [adj for adj in state.board.get_adjacents(maximum_pos[0], maximum_pos[1]) \
                            if state.board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], maximum + 1))
            if minimum - 1 in state.board.numbers_to_go:
                for pos in [adj for adj in state.board.get_adjacents(minimum_pos[0], minimum_pos[1]) \
                            if state.board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], minimum - 1))


        if len(state.board.islands) == 1:
            maximum =  max(islands[0])
            maximum_pos = state.board.assigned_positions[maximum]
            if maximum != state.board.board_size * state.board.board_size:
                for pos in [adj for adj in state.get_adjacents(maximum_pos[0], maximum_pos[1]) \
                            if state.board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], maximum + 1))

            minimum = min(islands[0])
            minimum_pos = state.board.assigned_positions[minimum]
            if minimum != 1:
                for pos in [adj for adj in state.get_adjacents(minimum_pos[0], minimum_pos[1]) \
                            if state.board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], minimum - 1))
        #print(f"{actions}")
        return actions


    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de 
        self.actions(state). """

        (row, col, value) = action
        #print(f"Before applying {action}")
        #print(state.board.islands)
        #print(state.board.island_map)
        #print(state.board.to_string())

        new_board = copy.copy(state.board)
        new_board.board_size = state.board.board_size
        new_board.board = copy.copy(state.board.board)
        new_board.islands = [copy.copy(island) for island in state.board.islands]
        new_board.island_map = {}
        for island in new_board.islands:
            for number in island:
                new_board.island_map[number] = island

        new_board.assigned_positions = copy.copy(state.board.assigned_positions)

        new_board.set_number(row, col, value)
        new_board.numbers_to_go = state.board.numbers_to_go - {value}
        new_board.assigned_positions[value] = (row, col)

        adj_neigh =  [adj for adj in new_board.get_adjacents(row, col) \
                if new_board.get_number(adj[0], adj[1]) != 0 and \
                abs(new_board.get_number(adj[0], adj[1]) - value) == 1][0]
        #print(f"My neighbor is {adj_neigh}")

        adj_neigh_island = new_board.island_map[new_board.get_number(adj_neigh[0], adj_neigh[1])]
        #print(f"His island is {adj_neigh_island}")
        adj_neigh_island.update(adj_neigh_island.union({value}))
        #print(f"After adding it becomes: {adj_neigh_island}")

        for pos in [adj for adj in new_board.get_adjacents(row, col) \
                      if new_board.get_number(adj[0], adj[1]) != 0 and \
                    abs(new_board.get_number(adj[0], adj[1]) - value) == 1 and \
                    adj != adj_neigh]:

            adj_island = new_board.island_map[new_board.get_number(pos[0], pos[1])]
            #print(f"Analyzing {adj_island}")
            #print(f"Against {adj_neigh_island}")
            adj_neigh_island.update(adj_neigh_island.union(adj_island))

            new_board.islands.remove(adj_island)

        #print(f"Applying {action}")
        #print(new_board.islands)
        #print(new_board.island_map)
        #print(new_board.to_string())
        return NumbrixState(new_board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. """
        return len(state.board.islands) == 1 and len(state.board.numbers_to_go) == 0

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        return node.state.board.manhattan_distances()

    # TODO: outros metodos da classe


if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    tic = time.perf_counter()
    problem = Numbrix(board)
    goal_node = greedy_search(problem)
    toc = time.perf_counter()
    print(f"Programa executado em {toc - tic:0.4f} segundos.")
    print(goal_node.state.board.to_string(), sep="")