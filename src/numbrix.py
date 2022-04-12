# numbrix.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 20:
# 95565 Duarte Almeida
# 95587 Gustavo Aguiar

import sys
from search import Problem, Node, astar_search, breadth_first_tree_search, depth_first_tree_search, greedy_search, \
    recursive_best_first_search

# Auxiliary Functions

def is_square(matrix):
    """ Auxiliary function that receives a matrix and returns true if it's squared, and false otherwise. """
    return all(len(row) == len(matrix) for row in matrix)

# --------------------

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

    def __init__(self, board_size, board):
        self.board_size = board_size
        self.board = board

    def get_number(self, row: int, col: int) -> int:
        """ Devolve o valor na respetiva posição do tabuleiro. """

        try:
            return self.board[row][col]
        except IndexError:
            return None

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente abaixo e acima,
        respectivamente. """

        result = []
        for direction in [1, -1]:
            try:
                result.append(self.board[row + direction][col])
            except IndexError:
                result.append(None)
        return tuple(result)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
        """ Devolve os valores imediatamente à esquerda e à direita,
        respectivamente. """

        result = []
        for direction in [-1, 1]:
            try:
                result.append(self.board[row][col + direction])
            except IndexError:
                result.append(None)
        return tuple(result)

    @staticmethod
    def parse_instance(filename: str):
        """ Lê o ficheiro cujo caminho é passado como argumento e retorna
        uma instância da classe Board. """

        file_board = []
        try:
            with open(filename) as f:
                board_size = f.readline()
                for line in f.readlines():
                    split = line.split('\t')
                    file_board.append([int(i) for i in split])
            if not is_square(file_board):
                raise ValueError
            return Board(board_size, file_board)
        except IOError:  # Couldn't open input file
            print("Something went wrong while attempting to read file.")
            sys.exit(-1)
        except ValueError:  # Invalid input board
            print("The given matrix isn't square (NxN).")
            f.close()
            sys.exit(-1)

    # TODO: outros metodos da classe

    def to_string(self):
        return '\n'.join('\t'.join(f'{i}' for i in x) for x in self.board)


class Numbrix(Problem):
    def __init__(self, board: Board):
        """ O construtor especifica o estado inicial. """
        # TODO
        pass

    def actions(self, state: NumbrixState):
        """ Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. """
        # TODO
        pass

    def result(self, state: NumbrixState, action):
        """ Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state). """
        # TODO
        pass

    def goal_test(self, state: NumbrixState):
        """ Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes. """
        # TODO
        pass

    def h(self, node: Node):
        """ Função heuristica utilizada para a procura A*. """
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass
