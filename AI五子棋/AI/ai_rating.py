# AI_part/ai_rating.py
BOARD_SIZE = 15                 # 棋盘大小常量
EMPTY = 0                       # 空位标识
BLACK = 1                       # 黑棋标识
WHITE = 2                       # 白棋标识

# 评分常量，用于评估棋型强度
SCORE_FIVE = 1000000            # 五子连珠
SCORE_LIVE_FOUR = 100000        # 活四（两端开放的四子）
SCORE_RUSH_FOUR = 50000         # 冲四（一端开放的四子）
SCORE_LIVE_THREE = 10000        # 活三
SCORE_SLEEP_THREE = 5000        # 眠三
SCORE_LIVE_TWO = 1000           # 活二
SCORE_SLEEP_TWO = 500           # 眠二
SCORE_LIVE_ONE = 100            # 活一
SCORE_SLEEP_ONE = 50            # 眠一

def evaluate_point(board, row, col, color):
    """
    评估在(row, col)处下color棋子的总分（四个方向累加）
    参数：
        board: 15x15二维列表
        row, col: 评估位置
        color: 棋子颜色
    返回：该位置的得分
    """
    if board[row][col] != EMPTY:   # 如果已有棋子，得分为0
        return 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个方向：水平、垂直、主对角线、副对角线
    total_score = 0                # 总分初始化

    for dr, dc in directions:      # 遍历每个方向
        count = 1                  # 当前棋子计数，包含自身
        open_ends = 0              # 两端开放的端口数（0,1,2）
        # 正方向延伸
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:  # 在棋盘范围内
            if board[r][c] == color:        # 同色棋子
                count += 1
            elif board[r][c] == EMPTY:      # 遇到空位，算一个开放端
                open_ends += 1
                break                       # 该方向停止计数
            else:                           # 对方棋子，阻塞
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

        # 根据连续棋子数和开放端数进行评分
        if count >= 5:                             # 形成五连或以上
            total_score += SCORE_FIVE
        elif count == 4:
            if open_ends >= 2:
                total_score += SCORE_LIVE_FOUR     # 活四
            elif open_ends == 1:
                total_score += SCORE_RUSH_FOUR     # 冲四
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
    参数：
        board: 当前棋盘
        ai_color: AI颜色
        player_color: 对手颜色（用于防守）
    返回：(row, col) 或 None（棋盘已满）
    """
    best_score = -1              # 初始化最佳分数
    best_move = None             # 最佳走法
    for r in range(BOARD_SIZE):   # 遍历所有行
        for c in range(BOARD_SIZE):
            if board[r][c] != EMPTY:   # 跳过非空位
                continue
            # 进攻分：假设AI下在此处
            attack = evaluate_point(board, r, c, ai_color)
            # 防守分：假设对手下在此处的威胁
            defense = evaluate_point(board, r, c, player_color)
            total = attack * 1.1 + defense  # 进攻权重稍高
            if total > best_score:
                best_score = total
                best_move = (r, c)
    return best_move
