import multiprocessing as mp
from functools import partial

# 常量定义
BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

# 评分常量
SCORE_FIVE = 1000000
SCORE_LIVE_FOUR = 100000
SCORE_RUSH_FOUR = 50000
SCORE_LIVE_THREE = 10000
SCORE_SLEEP_THREE = 5000
SCORE_LIVE_TWO = 1000
SCORE_SLEEP_TWO = 500
SCORE_LIVE_ONE = 100
SCORE_SLEEP_ONE = 50


def evaluate_point(board, row, col, color):
    """
    评估在(row, col)处下color棋子的总分（四个方向累加）
    返回该位置的得分
    """
    if board[row][col] != EMPTY:
        return 0

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    total_score = 0

    for dr, dc in directions:
        count = 1
        open_ends = 0

        # 正方向
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r][c] == color:
                count += 1
            elif board[r][c] == EMPTY:
                open_ends += 1
                break
            else:
                break
            r += dr
            c += dc

        # 反方向
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board[r][c] == color:
                count += 1
            elif board[r][c] == EMPTY:
                open_ends += 1
                break
            else:
                break
            r -= dr
            c -= dc

        # 根据连续棋子数和开放端评分
        if count >= 5:
            total_score += SCORE_FIVE
        elif count == 4:
            if open_ends >= 2:
                total_score += SCORE_LIVE_FOUR
            elif open_ends == 1:
                total_score += SCORE_RUSH_FOUR
        elif count == 3:
            if open_ends >= 2:
                total_score += SCORE_LIVE_THREE
            elif open_ends == 1:
                total_score += SCORE_SLEEP_THREE
        elif count == 2:
            if open_ends >= 2:
                total_score += SCORE_LIVE_TWO
            elif open_ends == 1:
                total_score += SCORE_SLEEP_TWO
        elif count == 1:
            if open_ends >= 2:
                total_score += SCORE_LIVE_ONE
            elif open_ends == 1:
                total_score += SCORE_SLEEP_ONE

    return total_score


def _score_for_position(board, ai_color, player_color, pos):
    """
    计算单个空位的总得分（供并行调用）
    pos: (row, col)
    """
    r, c = pos
    if board[r][c] != EMPTY:
        return pos, -1  # 无效位置返回负分

    attack = evaluate_point(board, r, c, ai_color)
    defense = evaluate_point(board, r, c, player_color)
    total = attack * 1.1 + defense
    return pos, total


def get_best_move(board, ai_color, player_color):
    """
    并行评分接口：返回最佳落子位置 (row, col)
    如果棋盘已满，返回 None
    """
    # 收集所有空位
    empty_positions = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
                       if board[r][c] == EMPTY]
    if not empty_positions:
        return None

    # 使用进程池并行计算
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # 固定 board 和颜色参数
        func = partial(_score_for_position, board, ai_color, player_color)
        results = pool.map(func, empty_positions)

    # 找出总分最高的位置（忽略无效位置）
    best_pos, best_score = max(results, key=lambda x: x[1])
    return best_pos