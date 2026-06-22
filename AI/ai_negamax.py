"""
Negamax 搜索（Alpha-Beta 剪枝 + 迭代加深 + 强制手延伸）五子棋 AI
- 强制手延伸：必胜点 / 必堵点不消耗深度（quiescence），天然支持 VCF 求解
- 整盘线段评估：逐方向扫描连续同色段，按棋型打分
"""
import time
import numpy as np
from numba import njit
from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE

# 评分常量
WIN_SCORE         = 10_000_000
SCORE_FIVE        = 1_000_000
SCORE_LIVE_FOUR   = 100_000
SCORE_RUSH_FOUR   = 50_000
SCORE_LIVE_THREE  = 10_000
SCORE_SLEEP_THREE = 5_000
SCORE_LIVE_TWO    = 1_000
SCORE_SLEEP_TWO   = 500
SCORE_LIVE_ONE    = 100
SCORE_SLEEP_ONE   = 50

# 搜索参数
MAX_DEPTH     = 6
TIME_LIMIT    = 2.5
CANDIDATE_K   = 12
NEARBY_RADIUS = 1

_INF = WIN_SCORE * 4


# ==================== JIT 热路径函数 ====================

@njit(cache=True)
def _check_win(board, row, col, color):
    """判断 (row,col) 落 color 子后是否五连"""
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    for di in range(4):
        dr = dirs[di][0]
        dc = dirs[di][1]
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15 and board[r, c] == color:
            count += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < 15 and 0 <= c < 15 and board[r, c] == color:
            count += 1
            r -= dr
            c -= dc
        if count >= 5:
            return True
    return False


@njit(cache=True)
def _seg_score(count, open_ends):
    """根据连续同色数与开放端数评分"""
    if count >= 5:
        return SCORE_FIVE
    if count == 4:
        if open_ends >= 2:
            return SCORE_LIVE_FOUR
        if open_ends == 1:
            return SCORE_RUSH_FOUR
        return 0
    if count == 3:
        if open_ends >= 2:
            return SCORE_LIVE_THREE
        if open_ends == 1:
            return SCORE_SLEEP_THREE
        return 0
    if count == 2:
        if open_ends >= 2:
            return SCORE_LIVE_TWO
        if open_ends == 1:
            return SCORE_SLEEP_TWO
        return 0
    if count == 1:
        if open_ends >= 2:
            return SCORE_LIVE_ONE
        if open_ends == 1:
            return SCORE_SLEEP_ONE
        return 0
    return 0


@njit(cache=True)
def _evaluate_board(board, color):
    """
    整盘静态评估，返回 color 视角分数（正=优势）。
    逐方向扫描每个连续同色段，每段只从起点计一次，避免重复。
    """
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    score = 0
    for di in range(4):
        dr = dirs[di][0]
        dc = dirs[di][1]
        for r in range(15):
            for c in range(15):
                v = board[r, c]
                if v == 0:
                    continue
                # 只从段起点统计：前一格不同色
                pr, pc = r - dr, c - dc
                if 0 <= pr < 15 and 0 <= pc < 15 and board[pr, pc] == v:
                    continue
                count = 0
                rr, cc = r, c
                while 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == v:
                    count += 1
                    rr += dr
                    cc += dc
                open_ends = 0
                if 0 <= pr < 15 and 0 <= pc < 15 and board[pr, pc] == 0:
                    open_ends += 1
                if 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == 0:
                    open_ends += 1
                s = _seg_score(count, open_ends)
                if v == color:
                    score += s
                else:
                    score -= s
    return score


@njit(cache=True)
def _evaluate_point(board, row, col, color):
    """单点评分：假设 color 落在 (row,col) 的四方向总分（走法排序用）"""
    if board[row, col] != 0:
        return 0
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    total = 0
    for di in range(4):
        dr = dirs[di][0]
        dc = dirs[di][1]
        count = 1
        open_ends = 0
        r, c = row + dr, col + dc
        while 0 <= r < 15 and 0 <= c < 15:
            if board[r, c] == color:
                count += 1
                r += dr
                c += dc
            elif board[r, c] == 0:
                open_ends += 1
                break
            else:
                break
        r, c = row - dr, col - dc
        while 0 <= r < 15 and 0 <= c < 15:
            if board[r, c] == color:
                count += 1
                r -= dr
                c -= dc
            elif board[r, c] == 0:
                open_ends += 1
                break
            else:
                break
        total += _seg_score(count, open_ends)
    return total


@njit(cache=True)
def _get_nearby_empty(board, radius):
    """返回有棋子邻居的空位 (rs, cs, n)"""
    rs = np.empty(225, dtype=np.int32)
    cs = np.empty(225, dtype=np.int32)
    n = 0
    has_stone = False
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                has_stone = True
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < 15 and 0 <= nc < 15 and board[nr, nc] == 0:
                            found = False
                            for i in range(n):
                                if rs[i] == nr and cs[i] == nc:
                                    found = True
                                    break
                            if not found:
                                rs[n] = nr
                                cs[n] = nc
                                n += 1
    if not has_stone:
        rs[0] = 7
        cs[0] = 7
        n = 1
    return rs, cs, n


@njit(cache=True)
def _scan_wins(board, color, rs, cs, n):
    """在给定空位中扫描 color 的必胜落子。返回 (first_r, first_c, count)"""
    fr = -1
    fc = -1
    cnt = 0
    for i in range(n):
        r = rs[i]
        c = cs[i]
        board[r, c] = color
        if _check_win(board, r, c, color):
            if fr < 0:
                fr = r
                fc = c
            cnt += 1
        board[r, c] = 0
    return fr, fc, cnt


@njit(cache=True)
def _score_candidates(board, color, rs, cs, n):
    """对每个空位计算进攻+防守综合分，返回并行数组"""
    opp = 3 - color
    out_r = np.empty(n, dtype=np.int32)
    out_c = np.empty(n, dtype=np.int32)
    out_s = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = rs[i]
        c = cs[i]
        atk = _evaluate_point(board, r, c, color)
        dfn = _evaluate_point(board, r, c, opp)
        out_r[i] = r
        out_c[i] = c
        out_s[i] = atk * 1.2 + dfn   # 进攻略高于防守
    return out_r, out_c, out_s


class _Timeout(Exception):
    """迭代加深超时异常"""
    pass


# ==================== Negamax 主体 ====================

def _negamax(board, depth, alpha, beta, color, deadline, ply):
    """
    Negamax + Alpha-Beta 剪枝，返回 color 视角的分数。超时抛出 _Timeout。
    强制手延伸：必胜/必堵判断先于深度检查，不消耗深度。
    """
    if time.time() > deadline:
        raise _Timeout()

    opp = 3 - color

    rs, cs, n = _get_nearby_empty(board, NEARBY_RADIUS)
    if n == 0:
        return 0

    # 1. 自己有必胜点 → 立即赢（越浅越好）
    myr, myc, _ = _scan_wins(board, color, rs, cs, n)
    if myr >= 0:
        return WIN_SCORE - ply

    # 2. 对手有必胜点 → 必须堵
    br, bc, bcnt = _scan_wins(board, opp, rs, cs, n)
    if bcnt >= 2:
        # 双必胜点，无法全堵，判负（越深越晚输）
        return -(WIN_SCORE - ply - 1)
    if bcnt == 1:
        board[br, bc] = color
        if _check_win(board, br, bc, color):
            board[br, bc] = 0
            return WIN_SCORE - ply
        score = -_negamax(board, depth - 1, -beta, -alpha, opp, deadline, ply + 1)
        board[br, bc] = 0
        return score

    # 3. 叶子节点：静态评估
    if depth <= 0:
        return _evaluate_board(board, color)

    # 4. 生成并排序候选走法
    cr, cc, cscore = _score_candidates(board, color, rs, cs, n)
    cand = [(int(cr[i]), int(cc[i]), float(cscore[i])) for i in range(n)]
    cand.sort(key=lambda x: x[2], reverse=True)
    cand = cand[:CANDIDATE_K]

    best = -_INF
    for r, c, _ in cand:
        board[r, c] = color
        score = -_negamax(board, depth - 1, -beta, -alpha, opp, deadline, ply + 1)
        board[r, c] = 0
        if score > best:
            best = score
            if best > alpha:
                alpha = best
                if alpha >= beta:
                    break   # β 剪枝
    return best


# ==================== 主入口 ====================

def get_best_move(board, ai_color, player_color):
    """Negamax 搜索入口，返回最佳落子 (row, col)"""
    nb = np.array(board, dtype=np.int8)

    # 空盘下天元
    if int(np.count_nonzero(nb)) == 0:
        return (BOARD_SIZE // 2, BOARD_SIZE // 2)

    rs, cs, n = _get_nearby_empty(nb, NEARBY_RADIUS)

    # 1. 自己一步赢
    myr, myc, _ = _scan_wins(nb, ai_color, rs, cs, n)
    if myr >= 0:
        return (int(myr), int(myc))

    # 2. 对手一步赢 → 必须堵
    br, bc, bcnt = _scan_wins(nb, player_color, rs, cs, n)
    if bcnt >= 1:
        return (int(br), int(bc))

    # 3. 候选走法排序
    cr, cc, cscore = _score_candidates(nb, ai_color, rs, cs, n)
    ordered = [(int(cr[i]), int(cc[i]), float(cscore[i])) for i in range(n)]
    ordered.sort(key=lambda x: x[2], reverse=True)
    ordered = ordered[:CANDIDATE_K]

    if not ordered:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if nb[r, c] == EMPTY:
                    return (r, c)
        return None

    best_move = ordered[0][:2]
    opp = player_color
    deadline = time.time() + TIME_LIMIT

    # 4. 迭代加深
    for depth in range(2, MAX_DEPTH + 1, 2):
        try:
            results = []
            alpha = -_INF
            beta = _INF
            # 上一轮最优走法排最前，加速剪枝
            ordered.sort(key=lambda x: 0 if x[:2] == best_move else 1)
            for r, c, _ in ordered:
                nb[r, c] = ai_color
                if _check_win(nb, r, c, ai_color):
                    nb[r, c] = 0
                    return (r, c)
                score = -_negamax(nb, depth - 1, -beta, -alpha, opp, deadline, 1)
                nb[r, c] = 0
                results.append((r, c, score))
                if score > alpha:
                    alpha = score
            # 按本轮得分重排，供下一轮使用
            results.sort(key=lambda x: x[2], reverse=True)
            ordered = results
            best_move = ordered[0][:2]
            best_score = ordered[0][2]
            # 找到必胜，提前结束
            if best_score >= WIN_SCORE - 1000:
                break
        except _Timeout:
            # 当前深度未完成，保留上一轮结果
            break

    return best_move
