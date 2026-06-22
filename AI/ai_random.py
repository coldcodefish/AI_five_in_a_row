import random
from .ai_rating import BOARD_SIZE, EMPTY

def get_best_move(board, ai_color, player_color):
    """随机走子：从所有空位中随机选择一个"""
    empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]
    return random.choice(empty_cells) if empty_cells else None
