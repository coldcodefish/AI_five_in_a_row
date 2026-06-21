# AI_part/ai_random.py
import random                      # 导入随机模块
from .ai_rating import BOARD_SIZE, EMPTY  # 从核心模块导入常量

def get_best_move(board, ai_color, player_color):
    """
    随机走子算法：从所有空位中随机选择一个
    参数与核心算法一致，但忽略颜色
    """
    # 收集所有空位坐标
    empty_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]
    # 如果存在空位，随机返回一个，否则返回None
    return random.choice(empty_cells) if empty_cells else None