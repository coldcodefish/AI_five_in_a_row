# AI/ai_mcts.py
import random
import math
import copy
from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE

SIMULATIONS = 2000
UCB_C = 1.4

class Node:
    def __init__(self, board, parent=None, move=None, color_of_last_move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.color_of_last_move = color_of_last_move
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self):
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == EMPTY]
        return len(self.children) == len(empty)

    def get_untried_moves(self):
        tried = [child.move for child in self.children]
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == EMPTY]
        return [move for move in empty if move not in tried]

    def best_child(self, c=UCB_C):
        # Prioritize unvisited children by giving them an infinite UCB score
        for child in self.children:
            if child.visits == 0:
                return child
        # Otherwise, use the UCB1 formula
        return max(self.children, key=lambda child: child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits))

def get_best_move(board, ai_color, player_color):
    root = Node(board=copy.deepcopy(board))
    for _ in range(SIMULATIONS):
        node = root
        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # Expansion
        untried = node.get_untried_moves()
        if untried:
            move = random.choice(untried)
            new_board = copy.deepcopy(node.board)
            # Calculate the player whose turn it is to move from the current node's board
            # This player will make the move 'move' to create the 'new_board'
            total_pieces_on_node_board = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if node.board[r][c] != EMPTY)
            player_making_this_move = BLACK if total_pieces_on_node_board % 2 == 0 else WHITE

            new_board[move[0]][move[1]] = player_making_this_move
            child = Node(board=new_board, parent=node, move=move, color_of_last_move=player_making_this_move)
            node.children.append(child)
            node = child
        # Simulation
        winner = simulate(node.board)
        # Backpropagation
        while node is not None:
            node.visits += 1
            # Only update wins if a player has made a move to reach this node and that player won
            if node.color_of_last_move is not None and node.color_of_last_move == winner:
                node.wins += 1
            node = node.parent

    if not root.children:
        return None
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move

def simulate(board):
    board_sim = [row[:] for row in board]  # 浅拷贝列表，比deepcopy快
    total = sum(1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board_sim[r][c] != EMPTY)
    current = BLACK if total % 2 == 0 else WHITE
    max_steps = BOARD_SIZE * BOARD_SIZE - total
    for _ in range(max_steps):
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board_sim[r][c] == EMPTY]
        if not empty:
            return None
        move = random.choice(empty)
        board_sim[move[0]][move[1]] = current
        if check_win(board_sim, move[0], move[1], current):
            return current
        current = WHITE if current == BLACK else BLACK
    return None

def check_win(board, row, col, color):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False