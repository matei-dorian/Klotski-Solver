import os
import sys

import stopit as stopit
from time import time
from queue import Queue, PriorityQueue
import heapq
from argparse import ArgumentParser


def copy_board(board):
    """
    Utility function that takes a board state as parameter
    and returns a new board with the same configuration
    """
    board_copy = [[x for x in lin] for lin in board]
    return board_copy


class SearchNode:
    """
    Class representing a node in the search tree built when we solve the problem
    Each node contains the following data:
        info - board configuration
        parent - reference to the parent node (or None for the root)
        cost - the cost to go from parent to current node
        h - heuristic factor
        f - value of the function used by the A* algorithm: f = g + h
        move - the direction of the move
        special_piece = (x1, y1, x2, y2) - coordinates of the special piece
        exiting = boolean - TRUE if a part of the special piece is outside the board
    """

    def __init__(self, info, parent=None, cost=0, h=0, move="", offset=(0, 0), exiting=False):
        self.info = info
        self.parent = parent
        self.g = cost if self.parent is None else (cost + self.parent.g)
        self.h = h
        self.f = self.g + self.h
        self.move = move
        if parent is None:
            self.special_piece = self.__find_special_piece()
        else:
            x1, y1, x2, y2 = parent.special_piece
            self.special_piece = (x1 + offset[0],
                                  y1 + offset[1],
                                  x2 + offset[0],
                                  y2 + offset[1])
        self.exiting = exiting

    def get_path(self):
        """
        Method that goes up on the search tree and returns a list representing
        the path that ends with the node that called the method
        """
        path = [self]
        node = self

        while node.parent is not None:
            path.insert(0, node.parent)
            node = node.parent

        return path

    def print_path(self):
        """
        This method uses the get_path method to print all the information about the path
        type of move + current state

        Returns: the length of the path (int), the full path (string)
        """
        path = self.get_path()
        text = ""
        idx = 1
        cost = 0
        for node in path:
            text += str(idx) + ")\n" + str(node) + "\n"
            cost = node.g
            idx += 1

        return cost, text

    def is_visited(self, new_node):
        """
        Method that checks if new_node represents a state that has already been visited
        node(self) - current state / new_node - the state we want to transition to

        Returns: True if new_node was already visited /  False otherwise
        """
        node = self
        while node is not None:
            if new_node.info == node.info:
                return True
            node = node.parent
        return False

    def __find_special_piece(self):
        """
        Method that finds the coordinates of the special block in a specific state
        Returns:
            A list of pairs representing th
            e coordinates of the special block
            (top-left, bottom-right corners)
        """
        x1, y1, x2, y2 = -1, -1, -1, -1  # (x1, y1) - top-left corner/ (x2, y2) - bottom-right corner
        for i in range(len(self.info)):
            for j in range(len(self.info[0])):
                if self.info[i][j] == '*':
                    if x1 == -1 and y1 == -1:
                        x1 = i
                        y1 = j
                    x2 = i
                    y2 = j
        return x1, y1, x2, y2

    def __lt__(self, obj):
        if self.f < obj.f:
            return 1
        if self.f == obj.f and self.g >= obj.g:
            return 1
        return 0

    def __hash__(self):
        return (tuple(line) for line in self.info).__hash__()

    def __eq__(self, obj):
        return self.info == obj.info

    def __str__(self):
        text = ""
        if self.move != "":
            text = self.move + '\n'
        text += '\n'.join([''.join([str(self.info[i][j]) for j in range(len(self.info[i]))])
                           for i in range(len(self.info))])
        return text + '\n'

    def __repr__(self):
        representation = str(self.info)
        return representation


class Graph:
    """
    The problem will be modeled using a graph
    The Graph class will contain:
        exit = tuple containing the coordinates of the exit
        obstacles = a dictionary where the key is the representation of the obstacle
                    (a letter) and the values are the number of blocks in the piece (cost to move) !!!!
        num_lin = number of lines in the board
        num_col = number of columns in the board
        nodes = a vector of boards

        a board = matrix of chars representing the state of the board
    """

    def __init__(self, filename):  # filename is the file from where we read the input
        f = open(filename)
        board = [[c for c in line.strip()] for line in f.readlines()]
        self.nodes = [SearchNode(board)]

        # check if the input is correct
        input_status, error_text = self.__validate_input()
        if not input_status:
            raise Exception(error_text)

        # the input is correct -> extract more info
        self.num_lin = len(board)
        self.num_col = len(board[0])

        # set the coordinates for the exit
        coord = [-1, -1]
        for j in range(self.num_col):
            if board[0][j] != '#':  # exit will always be at the top
                coord[1] = j
                if coord[0] == -1:
                    coord[0] = j
        self.exit = tuple(coord)
        self.obstacles = dict()

        for i in range(self.num_lin):
            for j in range(self.num_col):
                if board[i][j].isalpha():
                    # memorize obstacles
                    if board[i][j] not in self.obstacles.keys():
                        self.obstacles[board[i][j]] = 1
                    else:
                        self.obstacles[board[i][j]] += 1
        self.length = self.nodes[0].special_piece[2] - self.nodes[0].special_piece[0] + 1

    @staticmethod
    def __remove_piece(board, row, col):
        """
        Utility function that removes an obstacle from the board.
        This method is private because the user cannot remove pieces from the board.
        The method is used only for verifying the input data

        Args: the configuration of the board and the coordinates of the top-left corner of the piece
              that needs to be removed

        !!! this method is destructive - before calling this function make a copy of the board and pass
        it as a parameter
        """
        piece = board[row][col]
        q = Queue()
        q.put((row, col))
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # top, right, bottom, left
        n = len(board)  # number of rows
        m = len(board[0])  # number of columns

        while not q.empty():
            current_row, current_column = q.get()
            board[current_row][current_column] = '.'
            for direction in dirs:
                next_row = current_row + direction[0]
                next_column = current_column + direction[1]
                if 0 <= next_row <= (n - 1) and 0 <= next_column <= (m - 1):
                    if board[next_row][next_column] == piece:
                        q.put((next_row, next_column))

    @staticmethod
    def __get_move_text(symbol, dir1, dir2):
        """
        Args: symbol - the letter of the piece/ dir1, dir2 - offset of the row and column
        Return: A string that tells the user what piece was moved and in what direction
        """
        text = "Move " + symbol
        if dir1 == 0 and dir2 == -1:
            return text + " to the left"
        if dir1 == 0 and dir2 == 1:
            return text + " to the right"
        if dir1 == 1 and dir2 == 0:
            return text + " down"
        if dir1 == -1 and dir2 == 0:
            return text + " up"

    @staticmethod
    def check_final_state(state):
        """
        This method takes a board as parameter and checks whether the given board is a final state
        Returns: True if the board is a final state, False otherwise
        """
        x1, y1, x2, y2, = state.special_piece
        if x2 < 0:
            return True
        return False

    def __validate_input(self):
        """
        Method that checks for correct input data
        Returns:
            (True, "") if the input is correct
            (False, error_message) otherwise

        !!! this method will automatically make a copy of the board that will
        be used to call the __remove_piece method so that self is unmodified
        """
        board = self.nodes[0].info

        # check if the board has space inside it
        if len(board) < 3 or len(board[0]) < 3:
            return False, "Given board is too small"

        # check if all columns have the same length
        length = len(board[0])
        for row in board:
            if len(row) != length:
                return False, "Given board is not a rectangle"

        # check if the board contains only allowed characters
        unauthorized_characters = set()
        for row in board:
            for c in row:
                if c != '.' and c != '#' and c != '*' and not c.isalpha():
                    unauthorized_characters.add(c)
        if unauthorized_characters:
            return False, "Given board contains unauthorized characters: " + str(unauthorized_characters)

        # check for the margins to be filled with #
        for row in board[1:len(board) - 2]:
            if row[0] != '#' or row[length - 1] != '#':
                return False, "The margins of the board are not properly closed"

        for c in board[len(board) - 1]:
            if c != '#':
                return False, "The margins of the board are not properly closed"

        # check to have exactly one exit
        first_row = "".join(board[0]).strip('#')
        if first_row == "":
            return False, "Given board does not have an exit"

        for c in first_row:
            if c == '#':
                return False, "No more than one exit allowed"

        # check that every piece has a unique name and also that the special object exists
        pieces = set()
        board_copy = copy_board(board)

        for i in range(len(board)):
            for j in range(length):
                if board_copy[i][j].isalpha():
                    if board_copy[i][j] in pieces:
                        return False, "There cannot be two pieces with the same letter in the puzzle"
                    pieces.add(board_copy[i][j])
                    Graph.__remove_piece(board_copy, i, j)
                elif board_copy[i][j] == '*':
                    pieces.add(board_copy[i][j])

        if '*' not in pieces:
            return False, "The given puzzle is already in the finish state"

        # if we reached this point -> the board is valid
        return True, ""

    def impossible_to_solve(self):
        """
            Function that returns True if the final state cannot be reached from the start state
            Else returns False
        """
        board = self.nodes[0]
        piece_length = board.special_piece[3] - board.special_piece[1] + 1
        puzzle_length = self.num_col - 2
        max_length = puzzle_length - piece_length

        for i in range(board.special_piece[0]):
            current_obstacle = board.info[i][1]
            length = 1
            for j in range(2, self.num_col - 1):
                if board.info[i][j] == current_obstacle:
                    length += 1
                else:
                    if current_obstacle.isalpha() and length > max_length:
                        return True
                    current_obstacle = board.info[i][j]
                    length = 1
        return False

    def __move_piece(self, board, symbol, direction):
        """
        Args: board - current state of the puzzle
              symbol - the letter of the piece that will be moved
              direction - the direction in which we move the piece

        Returns: True, the new state of the board (cboard), cost of the move -> if the move was successful
                 False, None, 0 -> otherwise
        """
        cboard = copy_board(board)
        offset_x, offset_y = direction
        viz = [[False for _ in range(self.num_col)] for _ in range(self.num_lin)]

        for i in range(self.num_lin):
            for j in range(self.num_col):
                if board[i][j] == symbol:
                    next_x = i + offset_x
                    next_y = j + offset_y

                    if board[next_x][next_y] != '.' and board[next_x][next_y] != symbol:
                        return False, None, 0

                    if not viz[i][j]:
                        cboard[i][j] = '.'
                        viz[i][j] = True

                    cboard[next_x][next_y] = symbol
                    viz[next_x][next_y] = True

        cost = self.obstacles[symbol] if symbol != '*' else 1
        return True, cboard, cost

    def get_node_by_index(self, n):
        """
            This method returns the n-th node in the list of nodes
        """
        return self.nodes.index(n)

    def on_board(self, x, y):
        """
            Args: coordinates x (row), y (column)
            Returns: True -> if (x, y) position is on the board
                     False -> Otherwise
        """
        if 0 <= x < self.num_lin and 0 <= y < self.num_col:
            return True
        return False

    def manhattan_distance(self, node):
        """
            Args: a board
            Returns: minimum number of moves to get the special piece out of the board
                     (for this heuristic we will consider that there are no obstacles
                     on the board)
        """
        # finding the coordinates of the special piece
        special_coords = node.special_piece
        board = node.info

        if special_coords != (-1, -1, -1, -1):
            start_row, start_col, final_row, final_col = special_coords
            delta_row = final_row

            if self.exit[0] <= start_col and final_col <= self.exit[1]:
                # the special piece is aligned with the exit
                delta_col = 0
            elif final_col > self.exit[1]:
                # the special piece is on the right side of the exit
                delta_col = abs(final_col - self.exit[1])
            else:
                # the special piece is on the left side of the exit
                delta_col = abs(start_col - self.exit[0])

            unblock_exit = 0
            blocking_pieces = set()

            for i in range(self.exit[0], self.exit[1] + 1):
                if board[0][i] != '.' and board[0][i] != '*' and board[0][i] not in blocking_pieces:
                    blocking_pieces.add(board[0][i])
                    unblock_exit += self.obstacles[board[0][i]]

            return delta_row + delta_col + unblock_exit

        # if we called the method on a final state -> return 0
        return 0

    def estimate_number_of_steps(self, board, coords, offset):
        """
            This function calculates the minimum cost to move the special piece
        to the offset column then take it out of the puzzle. The estimation is done
        by counting the number of times we have to move the special piece + the cost of moving
        each obstacle only once (we can imagine that if encountered, an obstacle is removed)
        """
        cost = 0
        encountered_obstacles = set()
        direction = 1 if offset >= 0 else -1
        offset = abs(offset)
        x1, y1, x2, y2 = coords

        # move to the offset column
        for step in range(1, offset + 1):
            cost += 1  # move the special piece
            for k in range(x1, x2 + 1):
                position = board[k][y1 - step] if direction < 0 \
                    else board[k][y2 + step]
                if position.isalpha() and position not in encountered_obstacles:
                    encountered_obstacles.add(position)
                    cost += self.obstacles[position]

        if direction < 0:
            y1 -= offset
            y2 -= offset
        else:
            y1 += offset
            y2 += offset

        # move up
        while x1 > 1:
            cost += 1  # move the special piece
            x1 -= 1
            x2 -= 1
            for k in range(y1, y2 + 1):  # remove obstacles
                position = board[x1][k]
                if position.isalpha() and position not in encountered_obstacles:
                    encountered_obstacles.add(position)
                    cost += self.obstacles[position]

        # go to exit
        if y2 <= self.exit[1]:  # the piece is on the left side of the exit
            # move to the right
            while y2 != self.exit[1]:
                cost += 1
                y2 += 1
                for k in range(x1, x2 + 1):
                    position = board[k][y2]
                    if position.isalpha() and position not in encountered_obstacles:
                        encountered_obstacles.add(position)
                        cost += self.obstacles[position]
        else:
            # the piece is on the right side -> move to the left
            while y1 != self.exit[0]:
                cost += 1
                y1 -= 1
                for k in range(x1, x2 + 1):
                    position = board[k][y1]
                    if position.isalpha() and position not in encountered_obstacles:
                        encountered_obstacles.add(position)
                        cost += self.obstacles[position]

        # get the piece out of the puzzle
        for k in range(self.exit[0], self.exit[1] + 1):
            position = board[0][k]
            if position.isalpha() and position not in encountered_obstacles:
                encountered_obstacles.add(position)
                cost += self.obstacles[position]
        cost += x2 - x1 + 1
        return cost

    def admissible_heuristic2(self, node):
        """
            This function computes the heuristic factor using the estimate_number_of_steps function.
        The idea is to try every column and estimate the cost to the exit -> returns minimum cost
        """
        x1, y1, x2, y2 = node.special_piece  # x1, y1 - left-up corner, x2, y2 - right-down corner
        board = node.info
        coords = (x1, y1, x2, y2)
        evaluations = [self.estimate_number_of_steps(board, coords, 0)]
        offset = -1
        while board[x1][y1 + offset] != '#':  # go to the left as much as u can
            evaluations.append(self.estimate_number_of_steps(board, coords, offset))
            offset -= 1

        offset = 1
        while board[x1][y2 + offset] != '#':  # go to the right as much as u can
            evaluations.append(self.estimate_number_of_steps(board, coords, offset))
            offset += 1
        return min(evaluations)

    def calculate_h(self, node, heuristic="trivial"):
        if heuristic == "trivial":
            if Graph.check_final_state(node):
                return 0
            else:
                return 1

        if heuristic == "admissible1":  # Manhattan distance
            return self.manhattan_distance(node)

        if heuristic == "admissible2":  # Manhattan distance + row with the least number of obstacles
            return self.admissible_heuristic2(node)

        if heuristic == "non-admissible":
            return 2 * self.manhattan_distance(node) + 1

        return 0  # if heuristic does not have any of the value return 0 always

    def generate_successors(self, current_node, heuristic="trivial"):
        """
        This method expands the given node (current_node) and returns a list
        of possible next states

        Args: current_node - node to be expanded
              heuristic - the type of heuristic we want to use

        Return: list of the next board configurations
        """
        board = current_node.info
        list_suc = []
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # top, right, bottom, left
        symbols = set(self.obstacles.keys())
        # print(current_node)
        # print(current_node.exiting)
        if not current_node.exiting:
            symbols.add('*')
        #print(current_node.exiting)
        for obstacle in symbols:  # take every piece and try to move it in any direction
            for direction in dirs:
                status, new_state, cost = self.__move_piece(board, obstacle, direction)
                if status:
                    offset = direction if obstacle == '*' else (0, 0)
                    dummy = SearchNode(board, current_node, offset=offset)
                    h = self.calculate_h(dummy, heuristic)
                    move = Graph.__get_move_text(obstacle, *direction)
                    next_node = SearchNode(new_state, current_node, cost, h, move, offset)
                    if next_node.special_piece[0] <= 0:
                        next_node.exiting = True
                    if not current_node.is_visited(next_node):
                        list_suc.append(next_node)

        # state where the special piece leaves the puzzle
        lin_start, col_start, lin_end, col_end = current_node.special_piece

        if lin_start <= 0 and col_start >= self.exit[0] and col_end <= self.exit[1]:
            cboard = copy_board(board)
            for j in range(col_start, col_end + 1):
                cboard[lin_end][j] = '.'
            move = Graph.__get_move_text('*', -1, 0)
            next_node = SearchNode(cboard, current_node, 1, 0, move, (-1, 0), True)
            if not current_node.is_visited(next_node):
                list_suc.append(next_node)
        return list_suc

    def __repr__(self):
        text = ""
        for (k, v) in self.__dict__.items():
            text += "{}: {}\n".format(k, v)
        return text


def print_solution_in_file(file, node, total_time, max_nodes, num_computed_nodes):
    file.write("Solution:\n")
    cost, solution = node.print_path()
    file.write(solution)
    file.write("Total cost: " + str(cost) + '\n')
    file.write("Total time: " + str(total_time) + '\n')
    file.write("Total number of computed nodes: " + str(num_computed_nodes) + '\n')
    file.write("Maximum number of nodes in memory: " + str(max_nodes) + '\n')
    file.write("\n----------------\n")


@stopit.threading_timeoutable(default="Time limit exceeded")
def breadth_first(graph, file, num_sols=1):
    start_time = time()
    start_node = SearchNode(graph.nodes[0].info)
    q = Queue()
    q.put(start_node)
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while not q.empty():
        current_node = q.get()
        next_states_list = graph.generate_successors(current_node)
        num_of_computed_nodes += len(next_states_list)

        for state in next_states_list:
            if Graph.check_final_state(state):
                end_time = time()
                print_solution_in_file(file, state, round(end_time - start_time, 4), max_num_of_nodes,
                                       num_of_computed_nodes)
                num_sols -= 1
                if num_sols == 0:
                    return "Finished"
            q.put(state)
        max_num_of_nodes = max(max_num_of_nodes, q.qsize())
    return "Finished"


@stopit.threading_timeoutable(default="Time limit exceeded")
def depth_first(graph, file, num_sols=1):
    start_time = time()
    start_state = SearchNode(graph.nodes[0].info)
    max_num_of_nodes = 1
    num_of_computed_nodes = 1
    df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, start_state, num_sols)
    return "Finished"


def df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, current_node, num_sols=1):
    if num_sols <= 0:
        return num_sols
    max_num_of_nodes += 1
    if Graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                               num_of_computed_nodes)
        num_sols -= 1
        if num_sols == 0:
            return num_sols
    else:
        successors = graph.generate_successors(current_node)
        num_of_computed_nodes += len(successors)
        max_num_of_nodes = max(max_num_of_nodes, len(successors))
        for sc in successors:
            if num_sols != 0:
                num_sols = df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, sc, num_sols)

    return num_sols


@stopit.threading_timeoutable(default="Time limit exceeded")
def iterative_depth_first(graph, file, num_sol=1):
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1
    i = 1
    while num_sol != 0:
        num_sol = idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, graph.nodes[0], i, num_sol)
        i += 1
    return "Finished"


def idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, current_node, height, num_sol=1):
    num_of_computed_nodes += 1
    if height == 1 and Graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                               num_of_computed_nodes)
        num_sol -= 1
        if num_sol == 0:
            return num_sol
    if height > 1:
        successors = graph.generate_successors(current_node)
        num_of_computed_nodes += len(successors)
        max_num_of_nodes = max(max_num_of_nodes, len(successors))
        for sc in successors:
            if num_sol != 0:
                num_sol = idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, sc, height - 1, num_sol)
    return num_sol


@stopit.threading_timeoutable(default="Time limit exceeded")
def a_star(graph, file, num_sol=1, heuristic="trivial"):
    q = PriorityQueue()
    q.put(SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic)))
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while not q.empty():
        current_node = q.get()
        if graph.check_final_state(current_node):
            end_time = time()
            print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                                   num_of_computed_nodes)
            num_sol -= 1
            if num_sol == 0:
                return "Finished"

        successors = graph.generate_successors(current_node, heuristic)
        num_of_computed_nodes += len(successors)

        for successor in successors:
            if any(successor == item for item in q.queue):
                continue
            q.put(successor)
        max_num_of_nodes = max(max_num_of_nodes, q.qsize())


@stopit.threading_timeoutable(default="Time limit exceeded")
def a_star_opt(graph, file,  heuristic='trivial'):

    open_l = [SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic))]
    closed_l = dict()
    lazy_open = set()
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while len(open_l) > 0:
        current_node = heapq.heappop(open_l)
        while current_node in lazy_open:
            current_node = heapq.heappop(open_l)

        closed_l[current_node] = current_node.f
        if graph.check_final_state(current_node):
            end_time = time()
            print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                                   num_of_computed_nodes)
            return "Finished"

        successors = graph.generate_successors(current_node, heuristic)
        num_of_computed_nodes += len(successors)

        i = 0
        while i < len(successors):
            successor = successors[i]
            i += 1
            found = False
            for node in open_l:
                if successor == node:
                    found = True
                    if successor.f < node.f:
                        lazy_open.add(node)
                    else:
                        try:
                            successors.remove(successor)
                            i -= 1
                        except ValueError:
                            pass
            if not found:
                if successor in closed_l.keys():
                    if successor.f < closed_l[successor]:
                        del(closed_l[successor])
                    else:
                        try:
                            successors.remove(successor)
                            i -= 1
                        except ValueError:
                            pass

        for successor in successors:
            heapq.heappush(open_l, successor)
        max_num_of_nodes = max(max_num_of_nodes, len(open_l) + len(closed_l))


@stopit.threading_timeoutable(default="Time limit exceeded")
def ida_star(graph, file, num_sol=1, heuristic='trivial'):
    start_node = SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic))
    limit = start_node.f
    start_time = time()
    num_of_computed_nodes = 1
    max_num_of_nodes = 1

    while True:
        _, result = construct_path(graph, file, start_node, limit, num_sol, start_time, heuristic,
                                   num_of_computed_nodes, max_num_of_nodes)
        limit = result
        if result == "done":
            return "Finished"

        if result == float('inf'):
            file.write("No more solutions")
            return "Finished"


def construct_path(graph, file, current_node, limit, num_sol, start_time, heuristic, comp_nodes, max_nodes):
    if current_node.f > limit:
        return num_sol, current_node.f

    if graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_nodes,
                               comp_nodes)
        num_sol -= 1
        if num_sol == 0:
            return 0, "done"

    successors = graph.generate_successors(current_node, heuristic)
    comp_nodes += len(successors)
    max_nodes = max(max_nodes, len(successors))
    mini = float('inf')

    for successor in successors:
        num_sol, res = construct_path(graph, file, successor, limit, num_sol, start_time, heuristic,
                                      comp_nodes, max_nodes)

        if res == "done":
            return 0, "done"
        if res < mini:
            mini = res

    return num_sol, mini


def perform_all_algorithms(file, graph, nsol, timeout, heuristic):
    if graph.impossible_to_solve():
        file.write("The given puzzle is impossible to solve")
        return

    file.write("_________________ Breath First Search Algorithm _________________\n")
    status = breadth_first(graph, file, nsol, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of BFS _________________\n")

    file.write("\n__________________ Depth First Search Algorithm _________________\n")
    try:
        status = depth_first(graph, file, nsol, timeout=timeout)
        file.write(status + '\n')
    except:
        file.write("Maximum recursion depth exceeded\n")
    file.write("_________________ End of DFS _________________\n")

    file.write("\n_________________ Iterative Depth First Search Algorithm _________________\n")
    status = iterative_depth_first(graph, file, nsol, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of IDFS _________________\n")

    file.write("\n_________________ A* Algorithm _________________\n")
    status = a_star(graph, file, nsol, heuristic, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of A* _________________\n")

    file.write("\n_________________ Optimized A* Algorithm _________________\n")
    status = a_star_opt(graph, file, heuristic, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of Opt A* _________________\n")

    file.write("\n_________________ Iterative A* Algorithm _________________\n")
    try:
        status = ida_star(graph, file, nsol, heuristic, timeout=timeout)
        file.write(status + '\n')
    except:
        file.write("Maximum recursion depth exceeded\n")
    file.write("_________________ End of IDA* _________________\n")


if __name__ == "__main__":
    sys.setrecursionlimit(1500)
    # setting the arguments to run the program
    parser = ArgumentParser(description="Klotski puzzle solver")
    parser.add_argument("-if", "--input_folder",
                        dest="input_folder",
                        help="The path of the folder with the input files")
    parser.add_argument("-of", "--output_folder",
                        dest="output_folder",
                        help="The path of the folder with the output files")
    parser.add_argument("-nsol",
                        dest="nsol",
                        help="The number of solutions")
    parser.add_argument("-he", "--heuristic",
                        dest="heuristic",
                        help="The name of the heuristic function")
    parser.add_argument("-t", "--timeout",
                        dest="timeout",
                        help="The timeout for the searching algorithms")
    args = vars(parser.parse_args())

    # extracting the arguments
    input_folder_path = args["input_folder"]
    output_folder_path = args["output_folder"]
    nsol = int(args["nsol"])
    timeout = int(args["timeout"])
    heuristic = args["heuristic"]

    # create the output directory if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # get the files in the input directory
    files_list = os.listdir(input_folder_path)

    # add a / at the end to use the paths for the files
    if output_folder_path[-1] != '/':
        output_folder_path += '/'
    if input_folder_path[-1] != '/':
        input_folder_path += '/'

    # for each file in the input directory
    # run the algorithms and print the results in
    # the output directory in a separate file
    for (idx, file) in enumerate(files_list):
        output_file = "output" + str(idx + 1)
        i_file_path = input_folder_path + file
        o_file_path = output_folder_path + output_file

        f = open(o_file_path, "w")
        try:
            graph = Graph(i_file_path)
            perform_all_algorithms(f, graph, nsol, timeout, heuristic)
        except Exception as e:
            f.write(str(e))
        f.close()

import os
import sys

import stopit as stopit
from time import time
from queue import Queue, PriorityQueue
import heapq
from argparse import ArgumentParser


def copy_board(board):
    """
    Utility function that takes a board state as parameter
    and returns a new board with the same configuration
    """
    board_copy = [[x for x in lin] for lin in board]
    return board_copy


class SearchNode:
    """
    Class representing a node in the search tree built when we solve the problem
    Each node contains the following data:
        info - board configuration
        parent - reference to the parent node (or None for the root)
        cost - the cost to go from parent to current node
        h - heuristic factor
        f - value of the function used by the A* algorithm: f = g + h
        move - the direction of the move
        special_piece = (x1, y1, x2, y2) - coordinates of the special piece
        exiting = boolean - TRUE if a part of the special piece is outside the board
    """

    def __init__(self, info, parent=None, cost=0, h=0, move="", offset=(0, 0), exiting=False):
        self.info = info
        self.parent = parent
        self.g = cost if self.parent is None else (cost + self.parent.g)
        self.h = h
        self.f = self.g + self.h
        self.move = move
        if parent is None:
            self.special_piece = self.__find_special_piece()
        else:
            x1, y1, x2, y2 = parent.special_piece
            self.special_piece = (x1 + offset[0],
                                  y1 + offset[1],
                                  x2 + offset[0],
                                  y2 + offset[1])
        self.exiting = exiting

    def get_path(self):
        """
        Method that goes up on the search tree and returns a list representing
        the path that ends with the node that called the method
        """
        path = [self]
        node = self

        while node.parent is not None:
            path.insert(0, node.parent)
            node = node.parent

        return path

    def print_path(self):
        """
        This method uses the get_path method to print all the information about the path
        type of move + current state

        Returns: the length of the path (int), the full path (string)
        """
        path = self.get_path()
        text = ""
        idx = 1
        cost = 0
        for node in path:
            text += str(idx) + ")\n" + str(node) + "\n"
            cost = node.g
            idx += 1

        return cost, text

    def is_visited(self, new_node):
        """
        Method that checks if new_node represents a state that has already been visited
        node(self) - current state / new_node - the state we want to transition to

        Returns: True if new_node was already visited /  False otherwise
        """
        node = self
        while node is not None:
            if new_node.info == node.info:
                return True
            node = node.parent
        return False

    def __find_special_piece(self):
        """
        Method that finds the coordinates of the special block in a specific state
        Returns:
            A list of pairs representing th
            e coordinates of the special block
            (top-left, bottom-right corners)
        """
        x1, y1, x2, y2 = -1, -1, -1, -1  # (x1, y1) - top-left corner/ (x2, y2) - bottom-right corner
        for i in range(len(self.info)):
            for j in range(len(self.info[0])):
                if self.info[i][j] == '*':
                    if x1 == -1 and y1 == -1:
                        x1 = i
                        y1 = j
                    x2 = i
                    y2 = j
        return x1, y1, x2, y2

    def __lt__(self, obj):
        if self.f < obj.f:
            return 1
        if self.f == obj.f and self.g >= obj.g:
            return 1
        return 0

    def __hash__(self):
        return (tuple(line) for line in self.info).__hash__()

    def __eq__(self, obj):
        return self.info == obj.info

    def __str__(self):
        text = ""
        if self.move != "":
            text = self.move + '\n'
        text += '\n'.join([''.join([str(self.info[i][j]) for j in range(len(self.info[i]))])
                           for i in range(len(self.info))])
        return text + '\n'

    def __repr__(self):
        representation = str(self.info)
        return representation


class Graph:
    """
    The problem will be modeled using a graph
    The Graph class will contain:
        exit = tuple containing the coordinates of the exit
        obstacles = a dictionary where the key is the representation of the obstacle
                    (a letter) and the values are the number of blocks in the piece (cost to move) !!!!
        num_lin = number of lines in the board
        num_col = number of columns in the board
        nodes = a vector of boards

        a board = matrix of chars representing the state of the board
    """

    def __init__(self, filename):  # filename is the file from where we read the input
        f = open(filename)
        board = [[c for c in line.strip()] for line in f.readlines()]
        self.nodes = [SearchNode(board)]

        # check if the input is correct
        input_status, error_text = self.__validate_input()
        if not input_status:
            raise Exception(error_text)

        # the input is correct -> extract more info
        self.num_lin = len(board)
        self.num_col = len(board[0])

        # set the coordinates for the exit
        coord = [-1, -1]
        for j in range(self.num_col):
            if board[0][j] != '#':  # exit will always be at the top
                coord[1] = j
                if coord[0] == -1:
                    coord[0] = j
        self.exit = tuple(coord)
        self.obstacles = dict()

        for i in range(self.num_lin):
            for j in range(self.num_col):
                if board[i][j].isalpha():
                    # memorize obstacles
                    if board[i][j] not in self.obstacles.keys():
                        self.obstacles[board[i][j]] = 1
                    else:
                        self.obstacles[board[i][j]] += 1
        self.length = self.nodes[0].special_piece[2] - self.nodes[0].special_piece[0] + 1

    @staticmethod
    def __remove_piece(board, row, col):
        """
        Utility function that removes an obstacle from the board.
        This method is private because the user cannot remove pieces from the board.
        The method is used only for verifying the input data

        Args: the configuration of the board and the coordinates of the top-left corner of the piece
              that needs to be removed

        !!! this method is destructive - before calling this function make a copy of the board and pass
        it as a parameter
        """
        piece = board[row][col]
        q = Queue()
        q.put((row, col))
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # top, right, bottom, left
        n = len(board)  # number of rows
        m = len(board[0])  # number of columns

        while not q.empty():
            current_row, current_column = q.get()
            board[current_row][current_column] = '.'
            for direction in dirs:
                next_row = current_row + direction[0]
                next_column = current_column + direction[1]
                if 0 <= next_row <= (n - 1) and 0 <= next_column <= (m - 1):
                    if board[next_row][next_column] == piece:
                        q.put((next_row, next_column))

    @staticmethod
    def __get_move_text(symbol, dir1, dir2):
        """
        Args: symbol - the letter of the piece/ dir1, dir2 - offset of the row and column
        Return: A string that tells the user what piece was moved and in what direction
        """
        text = "Move " + symbol
        if dir1 == 0 and dir2 == -1:
            return text + " to the left"
        if dir1 == 0 and dir2 == 1:
            return text + " to the right"
        if dir1 == 1 and dir2 == 0:
            return text + " down"
        if dir1 == -1 and dir2 == 0:
            return text + " up"

    @staticmethod
    def check_final_state(state):
        """
        This method takes a board as parameter and checks whether the given board is a final state
        Returns: True if the board is a final state, False otherwise
        """
        x1, y1, x2, y2, = state.special_piece
        if x2 < 0:
            return True
        return False

    def __validate_input(self):
        """
        Method that checks for correct input data
        Returns:
            (True, "") if the input is correct
            (False, error_message) otherwise

        !!! this method will automatically make a copy of the board that will
        be used to call the __remove_piece method so that self is unmodified
        """
        board = self.nodes[0].info

        # check if the board has space inside it
        if len(board) < 3 or len(board[0]) < 3:
            return False, "Given board is too small"

        # check if all columns have the same length
        length = len(board[0])
        for row in board:
            if len(row) != length:
                return False, "Given board is not a rectangle"

        # check if the board contains only allowed characters
        unauthorized_characters = set()
        for row in board:
            for c in row:
                if c != '.' and c != '#' and c != '*' and not c.isalpha():
                    unauthorized_characters.add(c)
        if unauthorized_characters:
            return False, "Given board contains unauthorized characters: " + str(unauthorized_characters)

        # check for the margins to be filled with #
        for row in board[1:len(board) - 2]:
            if row[0] != '#' or row[length - 1] != '#':
                return False, "The margins of the board are not properly closed"

        for c in board[len(board) - 1]:
            if c != '#':
                return False, "The margins of the board are not properly closed"

        # check to have exactly one exit
        first_row = "".join(board[0]).strip('#')
        if first_row == "":
            return False, "Given board does not have an exit"

        for c in first_row:
            if c == '#':
                return False, "No more than one exit allowed"

        # check that every piece has a unique name and also that the special object exists
        pieces = set()
        board_copy = copy_board(board)

        for i in range(len(board)):
            for j in range(length):
                if board_copy[i][j].isalpha():
                    if board_copy[i][j] in pieces:
                        return False, "There cannot be two pieces with the same letter in the puzzle"
                    pieces.add(board_copy[i][j])
                    Graph.__remove_piece(board_copy, i, j)
                elif board_copy[i][j] == '*':
                    pieces.add(board_copy[i][j])

        if '*' not in pieces:
            return False, "The given puzzle is already in the finish state"

        # if we reached this point -> the board is valid
        return True, ""

    def impossible_to_solve(self):
        """
            Function that returns True if the final state cannot be reached from the start state
            Else returns False
        """
        board = self.nodes[0]
        piece_length = board.special_piece[3] - board.special_piece[1] + 1
        puzzle_length = self.num_col - 2
        max_length = puzzle_length - piece_length

        for i in range(board.special_piece[0]):
            current_obstacle = board.info[i][1]
            length = 1
            for j in range(2, self.num_col - 1):
                if board.info[i][j] == current_obstacle:
                    length += 1
                else:
                    if current_obstacle.isalpha() and length > max_length:
                        return True
                    current_obstacle = board.info[i][j]
                    length = 1
        return False

    def __move_piece(self, board, symbol, direction):
        """
        Args: board - current state of the puzzle
              symbol - the letter of the piece that will be moved
              direction - the direction in which we move the piece

        Returns: True, the new state of the board (cboard), cost of the move -> if the move was successful
                 False, None, 0 -> otherwise
        """
        cboard = copy_board(board)
        offset_x, offset_y = direction
        viz = [[False for _ in range(self.num_col)] for _ in range(self.num_lin)]

        for i in range(self.num_lin):
            for j in range(self.num_col):
                if board[i][j] == symbol:
                    next_x = i + offset_x
                    next_y = j + offset_y

                    if board[next_x][next_y] != '.' and board[next_x][next_y] != symbol:
                        return False, None, 0

                    if not viz[i][j]:
                        cboard[i][j] = '.'
                        viz[i][j] = True

                    cboard[next_x][next_y] = symbol
                    viz[next_x][next_y] = True

        cost = self.obstacles[symbol] if symbol != '*' else 1
        return True, cboard, cost

    def get_node_by_index(self, n):
        """
            This method returns the n-th node in the list of nodes
        """
        return self.nodes.index(n)

    def on_board(self, x, y):
        """
            Args: coordinates x (row), y (column)
            Returns: True -> if (x, y) position is on the board
                     False -> Otherwise
        """
        if 0 <= x < self.num_lin and 0 <= y < self.num_col:
            return True
        return False

    def manhattan_distance(self, node):
        """
            Args: a board
            Returns: minimum number of moves to get the special piece out of the board
                     (for this heuristic we will consider that there are no obstacles
                     on the board)
        """
        # finding the coordinates of the special piece
        special_coords = node.special_piece
        board = node.info

        if special_coords != (-1, -1, -1, -1):
            start_row, start_col, final_row, final_col = special_coords
            delta_row = final_row

            if self.exit[0] <= start_col and final_col <= self.exit[1]:
                # the special piece is aligned with the exit
                delta_col = 0
            elif final_col > self.exit[1]:
                # the special piece is on the right side of the exit
                delta_col = abs(final_col - self.exit[1])
            else:
                # the special piece is on the left side of the exit
                delta_col = abs(start_col - self.exit[0])

            unblock_exit = 0
            blocking_pieces = set()

            for i in range(self.exit[0], self.exit[1] + 1):
                if board[0][i] != '.' and board[0][i] != '*' and board[0][i] not in blocking_pieces:
                    blocking_pieces.add(board[0][i])
                    unblock_exit += self.obstacles[board[0][i]]

            return delta_row + delta_col + unblock_exit

        # if we called the method on a final state -> return 0
        return 0

    def estimate_number_of_steps(self, board, coords, offset):
        """
            This function calculates the minimum cost to move the special piece
        to the offset column then take it out of the puzzle. The estimation is done
        by counting the number of times we have to move the special piece + the cost of moving
        each obstacle only once (we can imagine that if encountered, an obstacle is removed)
        """
        cost = 0
        encountered_obstacles = set()
        direction = 1 if offset >= 0 else -1
        offset = abs(offset)
        x1, y1, x2, y2 = coords

        # move to the offset column
        for step in range(1, offset + 1):
            cost += 1  # move the special piece
            for k in range(x1, x2 + 1):
                position = board[k][y1 - step] if direction < 0 \
                    else board[k][y2 + step]
                if position.isalpha() and position not in encountered_obstacles:
                    encountered_obstacles.add(position)
                    cost += self.obstacles[position]

        if direction < 0:
            y1 -= offset
            y2 -= offset
        else:
            y1 += offset
            y2 += offset

        # move up
        while x1 > 1:
            cost += 1  # move the special piece
            x1 -= 1
            x2 -= 1
            for k in range(y1, y2 + 1):  # remove obstacles
                position = board[x1][k]
                if position.isalpha() and position not in encountered_obstacles:
                    encountered_obstacles.add(position)
                    cost += self.obstacles[position]

        # go to exit
        if y2 <= self.exit[1]:  # the piece is on the left side of the exit
            # move to the right
            while y2 != self.exit[1]:
                cost += 1
                y2 += 1
                for k in range(x1, x2 + 1):
                    position = board[k][y2]
                    if position.isalpha() and position not in encountered_obstacles:
                        encountered_obstacles.add(position)
                        cost += self.obstacles[position]
        else:
            # the piece is on the right side -> move to the left
            while y1 != self.exit[0]:
                cost += 1
                y1 -= 1
                for k in range(x1, x2 + 1):
                    position = board[k][y1]
                    if position.isalpha() and position not in encountered_obstacles:
                        encountered_obstacles.add(position)
                        cost += self.obstacles[position]

        # get the piece out of the puzzle
        for k in range(self.exit[0], self.exit[1] + 1):
            position = board[0][k]
            if position.isalpha() and position not in encountered_obstacles:
                encountered_obstacles.add(position)
                cost += self.obstacles[position]
        cost += x2 - x1 + 1
        return cost

    def admissible_heuristic2(self, node):
        """
            This function computes the heuristic factor using the estimate_number_of_steps function.
        The idea is to try every column and estimate the cost to the exit -> returns minimum cost
        """
        x1, y1, x2, y2 = node.special_piece  # x1, y1 - left-up corner, x2, y2 - right-down corner
        board = node.info
        coords = (x1, y1, x2, y2)
        evaluations = [self.estimate_number_of_steps(board, coords, 0)]
        offset = -1
        while board[x1][y1 + offset] != '#':  # go to the left as much as u can
            evaluations.append(self.estimate_number_of_steps(board, coords, offset))
            offset -= 1

        offset = 1
        while board[x1][y2 + offset] != '#':  # go to the right as much as u can
            evaluations.append(self.estimate_number_of_steps(board, coords, offset))
            offset += 1
        return min(evaluations)

    def calculate_h(self, node, heuristic="trivial"):
        if heuristic == "trivial":
            if Graph.check_final_state(node):
                return 0
            else:
                return 1

        if heuristic == "admissible1":  # Manhattan distance
            return self.manhattan_distance(node)

        if heuristic == "admissible2":  # Manhattan distance + row with the least number of obstacles
            return self.admissible_heuristic2(node)

        if heuristic == "non-admissible":
            return 2 * self.manhattan_distance(node) + 1

        return 0  # if heuristic does not have any of the value return 0 always

    def generate_successors(self, current_node, heuristic="trivial"):
        """
        This method expands the given node (current_node) and returns a list
        of possible next states

        Args: current_node - node to be expanded
              heuristic - the type of heuristic we want to use

        Return: list of the next board configurations
        """
        board = current_node.info
        list_suc = []
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # top, right, bottom, left
        symbols = set(self.obstacles.keys())
        # print(current_node)
        # print(current_node.exiting)
        if not current_node.exiting:
            symbols.add('*')
        #print(current_node.exiting)
        for obstacle in symbols:  # take every piece and try to move it in any direction
            for direction in dirs:
                status, new_state, cost = self.__move_piece(board, obstacle, direction)
                if status:
                    offset = direction if obstacle == '*' else (0, 0)
                    dummy = SearchNode(board, current_node, offset=offset)
                    h = self.calculate_h(dummy, heuristic)
                    move = Graph.__get_move_text(obstacle, *direction)
                    next_node = SearchNode(new_state, current_node, cost, h, move, offset)
                    if next_node.special_piece[0] <= 0:
                        next_node.exiting = True
                    if not current_node.is_visited(next_node):
                        list_suc.append(next_node)

        # state where the special piece leaves the puzzle
        lin_start, col_start, lin_end, col_end = current_node.special_piece

        if lin_start <= 0 and col_start >= self.exit[0] and col_end <= self.exit[1]:
            cboard = copy_board(board)
            for j in range(col_start, col_end + 1):
                cboard[lin_end][j] = '.'
            move = Graph.__get_move_text('*', -1, 0)
            next_node = SearchNode(cboard, current_node, 1, 0, move, (-1, 0), True)
            if not current_node.is_visited(next_node):
                list_suc.append(next_node)
        return list_suc

    def __repr__(self):
        text = ""
        for (k, v) in self.__dict__.items():
            text += "{}: {}\n".format(k, v)
        return text


def print_solution_in_file(file, node, total_time, max_nodes, num_computed_nodes):
    file.write("Solution:\n")
    cost, solution = node.print_path()
    file.write(solution)
    file.write("Total cost: " + str(cost) + '\n')
    file.write("Total time: " + str(total_time) + '\n')
    file.write("Total number of computed nodes: " + str(num_computed_nodes) + '\n')
    file.write("Maximum number of nodes in memory: " + str(max_nodes) + '\n')
    file.write("\n----------------\n")


@stopit.threading_timeoutable(default="Time limit exceeded")
def breadth_first(graph, file, num_sols=1):
    start_time = time()
    start_node = SearchNode(graph.nodes[0].info)
    q = Queue()
    q.put(start_node)
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while not q.empty():
        current_node = q.get()
        next_states_list = graph.generate_successors(current_node)
        num_of_computed_nodes += len(next_states_list)

        for state in next_states_list:
            if Graph.check_final_state(state):
                end_time = time()
                print_solution_in_file(file, state, round(end_time - start_time, 4), max_num_of_nodes,
                                       num_of_computed_nodes)
                num_sols -= 1
                if num_sols == 0:
                    return "Finished"
            q.put(state)
        max_num_of_nodes = max(max_num_of_nodes, q.qsize())
    return "Finished"


@stopit.threading_timeoutable(default="Time limit exceeded")
def depth_first(graph, file, num_sols=1):
    start_time = time()
    start_state = SearchNode(graph.nodes[0].info)
    max_num_of_nodes = 1
    num_of_computed_nodes = 1
    df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, start_state, num_sols)
    return "Finished"


def df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, current_node, num_sols=1):
    if num_sols <= 0:
        return num_sols
    max_num_of_nodes += 1
    if Graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                               num_of_computed_nodes)
        num_sols -= 1
        if num_sols == 0:
            return num_sols
    else:
        successors = graph.generate_successors(current_node)
        num_of_computed_nodes += len(successors)
        max_num_of_nodes = max(max_num_of_nodes, len(successors))
        for sc in successors:
            if num_sols != 0:
                num_sols = df(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, sc, num_sols)

    return num_sols


@stopit.threading_timeoutable(default="Time limit exceeded")
def iterative_depth_first(graph, file, num_sol=1):
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1
    i = 1
    while num_sol != 0:
        num_sol = idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, graph.nodes[0], i, num_sol)
        i += 1
    return "Finished"


def idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, current_node, height, num_sol=1):
    num_of_computed_nodes += 1
    if height == 1 and Graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                               num_of_computed_nodes)
        num_sol -= 1
        if num_sol == 0:
            return num_sol
    if height > 1:
        successors = graph.generate_successors(current_node)
        num_of_computed_nodes += len(successors)
        max_num_of_nodes = max(max_num_of_nodes, len(successors))
        for sc in successors:
            if num_sol != 0:
                num_sol = idf(file, start_time, max_num_of_nodes, num_of_computed_nodes, graph, sc, height - 1, num_sol)
    return num_sol


@stopit.threading_timeoutable(default="Time limit exceeded")
def a_star(graph, file, num_sol=1, heuristic="trivial"):
    q = PriorityQueue()
    q.put(SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic)))
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while not q.empty():
        current_node = q.get()
        if graph.check_final_state(current_node):
            end_time = time()
            print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                                   num_of_computed_nodes)
            num_sol -= 1
            if num_sol == 0:
                return "Finished"

        successors = graph.generate_successors(current_node, heuristic)
        num_of_computed_nodes += len(successors)

        for successor in successors:
            if any(successor == item for item in q.queue):
                continue
            q.put(successor)
        max_num_of_nodes = max(max_num_of_nodes, q.qsize())


@stopit.threading_timeoutable(default="Time limit exceeded")
def a_star_opt(graph, file,  heuristic='trivial'):

    open_l = [SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic))]
    closed_l = dict()
    lazy_open = set()
    start_time = time()
    max_num_of_nodes = 1
    num_of_computed_nodes = 1

    while len(open_l) > 0:
        current_node = heapq.heappop(open_l)
        while current_node in lazy_open:
            current_node = heapq.heappop(open_l)

        closed_l[current_node] = current_node.f
        if graph.check_final_state(current_node):
            end_time = time()
            print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_num_of_nodes,
                                   num_of_computed_nodes)
            return "Finished"

        successors = graph.generate_successors(current_node, heuristic)
        num_of_computed_nodes += len(successors)

        i = 0
        while i < len(successors):
            successor = successors[i]
            i += 1
            found = False
            for node in open_l:
                if successor == node:
                    found = True
                    if successor.f < node.f:
                        lazy_open.add(node)
                    else:
                        try:
                            successors.remove(successor)
                            i -= 1
                        except ValueError:
                            pass
            if not found:
                if successor in closed_l.keys():
                    if successor.f < closed_l[successor]:
                        del(closed_l[successor])
                    else:
                        try:
                            successors.remove(successor)
                            i -= 1
                        except ValueError:
                            pass

        for successor in successors:
            heapq.heappush(open_l, successor)
        max_num_of_nodes = max(max_num_of_nodes, len(open_l) + len(closed_l))


@stopit.threading_timeoutable(default="Time limit exceeded")
def ida_star(graph, file, num_sol=1, heuristic='trivial'):
    start_node = SearchNode(graph.nodes[0].info, None, 0, graph.calculate_h(graph.nodes[0], heuristic))
    limit = start_node.f
    start_time = time()
    num_of_computed_nodes = 1
    max_num_of_nodes = 1

    while True:
        _, result = construct_path(graph, file, start_node, limit, num_sol, start_time, heuristic,
                                   num_of_computed_nodes, max_num_of_nodes)
        limit = result
        if result == "done":
            return "Finished"

        if result == float('inf'):
            file.write("No more solutions")
            return "Finished"


def construct_path(graph, file, current_node, limit, num_sol, start_time, heuristic, comp_nodes, max_nodes):
    if current_node.f > limit:
        return num_sol, current_node.f

    if graph.check_final_state(current_node):
        end_time = time()
        print_solution_in_file(file, current_node, round(end_time - start_time, 4), max_nodes,
                               comp_nodes)
        num_sol -= 1
        if num_sol == 0:
            return 0, "done"

    successors = graph.generate_successors(current_node, heuristic)
    comp_nodes += len(successors)
    max_nodes = max(max_nodes, len(successors))
    mini = float('inf')

    for successor in successors:
        num_sol, res = construct_path(graph, file, successor, limit, num_sol, start_time, heuristic,
                                      comp_nodes, max_nodes)

        if res == "done":
            return 0, "done"
        if res < mini:
            mini = res

    return num_sol, mini


def perform_all_algorithms(file, graph, nsol, timeout, heuristic):
    if graph.impossible_to_solve():
        file.write("The given puzzle is impossible to solve")
        return

    file.write("_________________ Breath First Search Algorithm _________________\n")
    status = breadth_first(graph, file, nsol, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of BFS _________________\n")

    file.write("\n__________________ Depth First Search Algorithm _________________\n")
    try:
        status = depth_first(graph, file, nsol, timeout=timeout)
        file.write(status + '\n')
    except:
        file.write("Maximum recursion depth exceeded\n")
    file.write("_________________ End of DFS _________________\n")

    file.write("\n_________________ Iterative Depth First Search Algorithm _________________\n")
    status = iterative_depth_first(graph, file, nsol, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of IDFS _________________\n")

    file.write("\n_________________ A* Algorithm _________________\n")
    status = a_star(graph, file, nsol, heuristic, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of A* _________________\n")

    file.write("\n_________________ Optimized A* Algorithm _________________\n")
    status = a_star_opt(graph, file, heuristic, timeout=timeout)
    file.write(status + '\n')
    file.write("_________________ End of Opt A* _________________\n")

    file.write("\n_________________ Iterative A* Algorithm _________________\n")
    try:
        status = ida_star(graph, file, nsol, heuristic, timeout=timeout)
        file.write(status + '\n')
    except:
        file.write("Maximum recursion depth exceeded\n")
    file.write("_________________ End of IDA* _________________\n")


if __name__ == "__main__":
    sys.setrecursionlimit(1500)
    # setting the arguments to run the program
    parser = ArgumentParser(description="Klotski puzzle solver")
    parser.add_argument("-if", "--input_folder",
                        dest="input_folder",
                        help="The path of the folder with the input files")
    parser.add_argument("-of", "--output_folder",
                        dest="output_folder",
                        help="The path of the folder with the output files")
    parser.add_argument("-nsol",
                        dest="nsol",
                        help="The number of solutions")
    parser.add_argument("-he", "--heuristic",
                        dest="heuristic",
                        help="The name of the heuristic function")
    parser.add_argument("-t", "--timeout",
                        dest="timeout",
                        help="The timeout for the searching algorithms")
    args = vars(parser.parse_args())

    # extracting the arguments
    input_folder_path = args["input_folder"]
    output_folder_path = args["output_folder"]
    nsol = int(args["nsol"])
    timeout = int(args["timeout"])
    heuristic = args["heuristic"]

    # create the output directory if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # get the files in the input directory
    files_list = os.listdir(input_folder_path)

    # add a / at the end to use the paths for the files
    if output_folder_path[-1] != '/':
        output_folder_path += '/'
    if input_folder_path[-1] != '/':
        input_folder_path += '/'

    # for each file in the input directory
    # run the algorithms and print the results in
    # the output directory in a separate file
    for (idx, file) in enumerate(files_list):
        output_file = "output" + str(idx + 1)
        i_file_path = input_folder_path + file
        o_file_path = output_folder_path + output_file

        f = open(o_file_path, "w")
        try:
            graph = Graph(i_file_path)
            perform_all_algorithms(f, graph, nsol, timeout, heuristic)
        except Exception as e:
            f.write(str(e))
        f.close()

