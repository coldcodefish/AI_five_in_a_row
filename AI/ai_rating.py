BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

# 棋型评分常量
SCORE_FIVE = 1000000            # 五子连珠
SCORE_LIVE_FOUR = 100000        # 活四
SCORE_RUSH_FOUR = 50000         # 冲四
SCORE_LIVE_THREE = 10000        # 活三
SCORE_SLEEP_THREE = 5000        # 眠三
SCORE_LIVE_TWO = 1000           # 活二
SCORE_SLEEP_TWO = 500           # 眠二
SCORE_LIVE_ONE = 100            # 活一
SCORE_SLEEP_ONE = 50            # 眠一

def evaluate_point(board, row, col, color):
    """评估在(row, col)处下color棋子的总分（四个方向累加）"""
    if board[row][col] != EMPTY:
        return 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    total_score = 0

    for dr, dc in directions:
        count = 1
        open_ends = 0
        # 正方向延伸
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
        # 反方向延伸
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

def get_best_move(board, ai_color, player_color):
    """
    评分算法入口：遍历所有空位，计算进攻分和防守分，选择总分最高的位置
    返回：(row, col) 或 None（棋盘已满）
    """
    best_score = -1
    best_move = None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:
                continue
            attack = evaluate_point(board, r, c, ai_color)
            defense = evaluate_point(board, r, c, player_color)
            total = attack * 1.1 + defense  # 进攻权重稍高
            if total > best_score:
                best_score = total
                best_move = (r, c)
    return best_move
