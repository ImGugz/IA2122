# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 20:
# 95565 Duarte Almeida
# 95587 Gustavo Aguiar

import sys
import copy
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search


def get_minmax_pos(minmax):
    """ Dado um tuplo ((x, y), valor) retorna a posição (x, y). """
    return minmax[0]


def get_minmax_value(minmax):
    """ Dado um tuplo ((x, y), valor) retorna o valor. """
    return minmax[1]


def manhattan_distance(pos1, pos2):
    """ Devolve a distância Manhattan de dois vetores. """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class NumbrixState:
    """Representação interna de um estado de um tabuleiro de Numbrix. """
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NumbrixState.state_id
        NumbrixState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id


class Board:
    """ Representação interna de um tabuleiro de Numbrix. """

    board = {}  # board[(i, j)] = Value
    board_size = 0  # board_size = N^2
    numbers_to_go = []  # numbers_to_go = S in [1, 2, ..., N^2] s.t if i in S, the board doesn't contain number i
    local_min = ()  # local_min = ((lm_x, lm_y), lm_val) belonging to the minimum value of the second island (if appl.)
    local_max = ()  # local_max = ((lM_x, lM_y), lm_VAL) belonging to the maximum value of the first island (if appl.)
    global_min = ()  # global_min = ((m_x, m_y), m_val) containing the current minimum position and value on the board
    global_max = ()  # global_max = ((M_x, M_y), M_val) containing the current maximum position and value on the board
    extremes = False  # extremes = True, means that we only have 1 island remaining and will try to fill the value edges
    free_spaces = []  # free_spaces = [(i,j) s.t that board[(i, j)] = 0]

    def __init__(self, board, board_size, initial_numbers):
        self.board_size = board_size
        min_initial = min(initial_numbers)
        max_initial = max(initial_numbers)
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[(i, j)] = board[i][j]
                if board[i][j] == min_initial:
                    self.global_min = ((i, j), min_initial)
                if board[i][j] == max_initial:
                    self.global_max = ((i, j), max_initial)
                if self.board[(i, j)] == 0:
                    self.free_spaces.append((i, j))

        self.numbers_to_go = list(
            set(i for i in range(1, self.board_size * self.board_size + 1)) - set(initial_numbers))

        # Through a DFS from the global minimum try to expand an island from it and store its max position and value
        self.local_max = self.island_dfs(self.global_min)
        # Arbitrary number to be compared to find to minimum of a second island
        self.local_min = ((0, 0), self.board_size ** 2 + 1)

        for i in range(self.board_size):
            for j in range(self.board_size):
                num = self.get_number(i, j)
                if self.local_max[1] < num < self.local_min[1]:
                    self.local_min = ((i, j), num)

    def get_adjacents(self, row, col):
        """ Devolve uma lista de posições adjacentes que respeitem os limites do tabuleiro. """
        return list(filter(lambda x: 0 <= x[0] < self.board_size and 0 <= x[1] < self.board_size
                           , [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]))

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """
        try:
            return self.board[row, col]
        except KeyError:
            return None

    def set_number(self, row: int, col: int, value):
        """ Define um dado valor numa dada posição do tabuleiro. """
        try:
            self.board[row, col] = value
        except KeyError:
            pass

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente abaixo e acima, respectivamente. """
        results = []
        for direction in [1, -1]:
            try:
                results.append(self.board[row + direction][col])
            except IndexError:
                results.append(None)
        return tuple(results)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente à esquerda e à direita, respectivamente. """
        results = []
        for direction in [-1, 1]:
            try:
                results.append(self.board[row][col + direction])
            except IndexError:
                results.append(None)
        return tuple(results)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna uma instância da classe Board. """
        initial_numbers = []
        file_board = []
        try:
            with open(filename) as f:
                board_size = int(f.readline())
                for line in f.readlines():
                    split = line.split('\t')
                    initial_numbers += [int(val) for val in split if int(val) != 0]
                    file_board.append([int(val) for val in split])
            # TODO: Verificar construtor do Board
            return Board(file_board, board_size, initial_numbers)
        except IOError:  # Couldn't open input file
            print("Something went wrong while attempting to read file.")
            sys.exit(-1)

    def island_dfs(self, origin):
        """ Faz uma procura em profundidade primeiro a partir dum valor, e devolve um tuplo com uma posição e valor
        t.q o valor dessa mesma posição é o maior dessa ilha. """
        pos = get_minmax_pos(origin)
        val = get_minmax_value(origin)
        max_local = (pos, val)
        queue = [pos]
        visited = set()
        while len(queue) > 0:
            u = queue[-1]
            visited.add(u)
            num = self.get_number(u[0], u[1])
            if num > max_local[1] and num != val:
                max_local = (u, num)
            # Get all assigned and unvisited neighbors s.t they're value neighbors (abs(diff(values)) = 1
            neighbors = [adj for adj in self.get_adjacents(u[0], u[1])
                         if self.get_number(adj[0], adj[1]) != 0
                         and abs(self.get_number(adj[0], adj[1]) - num) == 1
                         and adj not in visited]
            if len(neighbors) > 0:
                queue.append(neighbors[0])
            else:
                queue = queue[:-1]
        return max_local

    def to_string(self):
        """ Representa um tabuleiro de acordo com o modelo definido no enunciado. """
        return '\n'.join('\t'.join(f'{self.get_number(i, j)}' for j in range(self.board_size))
                         for i in range(self.board_size))

    def dfs(self, origin, dest):
        """ Devolve True se ao aplicar uma procura em profundidade primeiro dest é atingível a partir de origin. """
        queue = [origin]
        visited = set()
        while len(queue) > 0:
            u = queue[-1]
            visited.add(u)
            if u == dest:
                return True
            neighbors = [adj for adj in self.get_adjacents(u[0], u[1])
                         if (self.get_number(adj[0], adj[1]) == 0 or adj == dest)
                         and adj not in visited]
            if len(neighbors) > 0:
                queue.append(neighbors[0])
            else:
                queue = queue[:-1]
        return False

    def is_space_reachable(self, origin, length):
        """ Devolve True se a partir de origin temos um caminho de pelo menos length células do tabuleiro vazias. """
        count = -1
        queue = [origin]
        visited = set()
        while len(queue) > 0:
            u = queue[-1]
            if u not in visited:
                count += 1
                visited.add(u)
            if count == length:
                return True
            neighbors = [adj for adj in self.get_adjacents(u[0], u[1])
                         if (self.get_number(adj[0], adj[1]) == 0)
                         and adj not in visited]
            if len(neighbors) > 0:
                queue.append(neighbors[0])
            else:
                queue = queue[:-1]
        return False

    def check_free_spaces(self, start):
        """ Devolve True se a partir de uma simulação de atribuição de valor no tabuleiro não criamos dead spaces. """
        visited = set()  # Set containing all visited nodes
        processed = set()  # Set containing all processed nodes

        # For each space in the board's free spaces
        for space in self.free_spaces:
            # After we visit and process each node, add it to the processed set
            processed.update(processed.union(visited))
            # If it's been processed there's nothing left to do with this node
            if space in processed:
                continue
            # Reset visited nodes set
            visited = set()
            no_visited = 0  # Number of visited nodes
            bigger_count = 0  # Count of nodes with bigger values seen
            bigger_list = []  # List of nodes with bigger values seen
            stack = [space]
            seen_upper = False  # True if we've seen the global maximum
            seen_lower = False  # True if we've seen the global minimum
            while stack:
                u = stack[-1]
                if u in processed:  # Optimization
                    bigger_count = 2
                    break
                # If we haven't visited u
                if u not in visited:
                    # Add it to the visited set
                    visited.add(u)
                    # Increase number of visited nodes
                    no_visited += 1
                    # And get its adjacents
                    for n in [adj for adj in self.get_adjacents(u[0], u[1])]:
                        # For each adjacent we get its value
                        value_n = self.get_number(n[0], n[1])
                        # If it's a free space and it hasn't been visited yet add it to the stack
                        if value_n == 0 and n not in visited:
                            stack.append(n)
                        # If it's bigger than the value we started we want to make sure that we can connect
                        # 2 nodes (there must be a gap in between them)
                        if value_n >= start and not (len(bigger_list) == 1 and abs(
                                    self.get_number(n[0], n[1]) - self.get_number(bigger_list[0][0],
                                                                                  bigger_list[0][1])) == 1):
                            bigger_list.append(n)
                            bigger_count += 1
                        # If we've seen 2 bigger valid nodes than we don't need to check more adjacents
                        if bigger_count == 2:
                            break
                        # And if we've seen the global min or max, set the respective flag
                        if get_minmax_value(self.global_min) == value_n:
                            seen_lower = True
                        if get_minmax_value(self.global_max) == value_n:
                            seen_upper = True
                    # If we've seen 2 bigger valid nodes break the DFS
                    if bigger_count == 2:
                        break
                else:
                    stack = stack[:-1]

            # 1st condition: we can connect 2 bigger value nodes
            # 2nd condition: we've seen global_min, and we can connect it with self.global_min (value) - 1 nodes
            # 3rd condition: we've seen global_max, and we can connect it with N^2 - self.global_max nodes
            # 4th condition: we've seen both global_min and global_max, and we can connect them
            if not (bigger_count == 2 or (seen_lower and no_visited == self.global_min[1] - 1) or
                    (seen_upper and no_visited == self.board_size ** 2 - self.global_max[1]) or
                    (seen_upper and seen_lower and (
                            no_visited == self.global_min[1] - 1 + self.board_size ** 2 - self.global_max[1]))):
                return False

        return True

    def __copy__(self):
        """ Devolve uma cópia vazia da classe Board. """
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
        """ Retorna uma lista de ações que podem ser executadas a partir do estado passado como argumento. """
        actions = []
        board = state.board

        global_min_pos = get_minmax_pos(board.global_min)
        global_min_val = get_minmax_value(board.global_min)
        global_max_pos = get_minmax_pos(board.global_max)
        global_max_val = get_minmax_value(board.global_max)

        # Check if we can continue the board from the lowest and highest occupied position
        if not (board.is_space_reachable(global_min_pos, global_min_val - 1)
                and board.is_space_reachable(global_max_pos, board.board_size ** 2 - global_max_val)):
            return []

        # There's more than an island
        if not board.extremes:

            local_max_pos = get_minmax_pos(board.local_max)
            local_max_val = get_minmax_value(board.local_max)
            local_min_pos = get_minmax_pos(board.local_min)
            local_min_val = get_minmax_value(board.local_min)

            # If the value distance between the 2 smallest islands is < than manhattan, no solution
            if manhattan_distance(local_max_pos, local_min_pos) > local_min_val - local_max_val:
                return []

            # If the 2 smallest islands aren't reacheable, no solution
            if not board.dfs(local_max_pos, local_min_pos):
                return []

            # Our goal is to always connect the 2 smallest islands from the smallest one to the second smallest
            if local_max_val + 1 in board.numbers_to_go:
                # For each free adjacent to the max
                for pos in [adj for adj in board.get_adjacents(local_max_pos[0], local_max_pos[1]) \
                            if board.get_number(adj[0], adj[1]) == 0]:
                    # We simulate its assignment
                    board.set_number(pos[0], pos[1], local_max_val + 1)
                    board.free_spaces.remove((pos[0], pos[1]))

                    # If the value difference is smaller than their manhattan distance, no solution
                    if manhattan_distance(pos, local_min_pos) > local_min_val - (local_max_val + 1):
                        board.free_spaces.append((pos[0], pos[1]))
                        board.set_number(pos[0], pos[1], 0)
                        continue

                    # If they're not reacheable, no solution
                    if not board.dfs(pos, local_min_pos):
                        board.free_spaces.append((pos[0], pos[1]))
                        board.set_number(pos[0], pos[1], 0)
                        continue

                    # If the free spaces are valid with this assignment, then append this assignment
                    if board.check_free_spaces(local_max_val + 1):
                        actions.append((pos[0], pos[1], local_max_val + 1))

                    # Remove the simulated assignment
                    board.free_spaces.append((pos[0], pos[1]))
                    board.set_number(pos[0], pos[1], 0)

        # We've reached a point where there's only 1 island S
        if board.extremes:
            # And we want to possibly complete {1, ..., min(S)}
            if global_max_val != board.board_size ** 2:
                for pos in [adj for adj in board.get_adjacents(global_max_pos[0], global_max_pos[1])
                            if board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], global_max_val + 1))
            # And to also complete {max(S), ..., N^2}
            if global_min_val != 1:
                for pos in [adj for adj in board.get_adjacents(global_min_pos[0], global_min_pos[1])
                            if board.get_number(adj[0], adj[1]) == 0]:
                    actions.append((pos[0], pos[1], global_min_val - 1))
        return actions

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """

        board = state.board

        # Unpack the action
        (row, col, value) = action
        # Copy the board and it's attributes accordingly
        new_board = copy.copy(board)
        new_board.board_size = board.board_size
        new_board.board = copy.copy(board.board)
        new_board.set_number(row, col, value)
        new_board.numbers_to_go = [x for x in board.numbers_to_go if x != value]
        new_board.free_spaces = [x for x in board.free_spaces if x != (row, col)]

        # Update global minimum if necessary
        if value < get_minmax_value(board.global_min):
            new_board.global_min = ((row, col), value)
        else:
            new_board.global_min = state.board.global_min

        # Update global maximum if necessary
        if value > get_minmax_value(board.global_max):
            new_board.global_max = ((row, col), value)
        else:
            new_board.global_max = board.global_max

        # Update locals before checking island convergency
        new_board.local_max = ((row, col), value)
        new_board.local_min = board.local_min
        if value + 1 == board.local_min[1]:  # Islands converged
            # We get the max value from an island DFS
            new_board.local_max = new_board.island_dfs(((row, col), value))
            # If we've reached the global max then we only have one island
            if new_board.local_max == new_board.global_max:
                new_board.local_min = new_board.global_min
            else:  # We want to check the min of the second-smallest island
                new_board.local_min = ((0, 0), state.board.board_size ** 2 + 1)
                for i in range(state.board.board_size):
                    for j in range(state.board.board_size):
                        num = new_board.get_number(i, j)
                        if new_board.local_max[1] < num < new_board.local_min[1]:
                            new_board.local_min = ((i, j), num)

        # There's only one island left
        if new_board.global_min == new_board.local_min or new_board.global_max == new_board.local_max:
            new_board.extremes = True

        return NumbrixState(new_board)

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro 
        estão preenchidas com uma sequência de números adjacentes. """
        return state.board.extremes and len(state.board.numbers_to_go) == 0

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        board = node.state.board
        h_n = 0
        # The idea is to minimize "holes"
        for space in board.free_spaces:  # For each free space check the number of free adjacents
            free_adj = [adj for adj in board.get_adjacents(space[0], space[1]) if board.get_number(adj[0], adj[1]) == 0]
            h_n += (4 - len(free_adj)) ** 1.4  # Exponent determined experimentally
        return h_n


if __name__ == "__main__":
    board = Board.parse_instance(sys.argv[1])
    problem = Numbrix(board)
    goal_node = greedy_search(problem)
    print(goal_node.state.board.to_string(), sep="")
