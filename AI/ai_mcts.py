import random
import math
import numpy as np
from numba import njit

from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE

# 超参数
SIMULATIONS = 600
UCB_C = 1.414              # sqrt(2)
TOP_K = 12
SIM_MAX_STEPS = 8

# ==================== Numba JIT 热路径函数 ====================

@njit(cache=True)
def _check_win(board, row, col, color):
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
def _evaluate_point(board, row, col, color):
    """单点评分：假设 color 落在 (row,col) 的四方向总分"""
    if board[row, col] != 0:
        return 0
    dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
    total_score = 0
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
        if count >= 5:
            total_score += 1000000
        elif count == 4:
            if open_ends >= 2:
                total_score += 100000
            elif open_ends == 1:
                total_score += 50000
        elif count == 3:
            if open_ends >= 2:
                total_score += 10000
            elif open_ends == 1:
                total_score += 5000
        elif count == 2:
            if open_ends >= 2:
                total_score += 1000
            elif open_ends == 1:
                total_score += 500
        elif count == 1:
            if open_ends >= 2:
                total_score += 100
            elif open_ends == 1:
                total_score += 50
    return total_score


@njit(cache=True)
def _get_nearby_moves(board, radius=2):
    """返回已有棋子周围 radius 范围内的空位"""
    moves_r = np.empty(225, dtype=np.int32)
    moves_c = np.empty(225, dtype=np.int32)
    count = 0
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
                            for i in range(count):
                                if moves_r[i] == nr and moves_c[i] == nc:
                                    found = True
                                    break
                            if not found:
                                moves_r[count] = nr
                                moves_c[count] = nc
                                count += 1
    if not has_stone:
        moves_r[0] = 7
        moves_c[0] = 7
        return moves_r, moves_c, 1
    return moves_r, moves_c, count


@njit(cache=True)
def _get_forced_defense(board, player_color):
    """对手落子即赢时返回位置，否则返回 (-1,-1)"""
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            board[r, c] = player_color
            if _check_win(board, r, c, player_color):
                board[r, c] = 0
                return r, c
            board[r, c] = 0
    return -1, -1


@njit(cache=True)
def _get_tactical_move(board, ai_color, player_color):
    """
    战术检查（在 MCTS 之前执行，处理明显威胁）：
    1. 自己能形成活四(100000分) → 直接走
    2. 对手能形成活四 → 堵
    3. 自己能形成双活三(两个方向≥10000) → 走
    4. 对手能形成双活三 → 堵
    """
    # 1. 自己能形成活四或五连
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            s = _evaluate_point(board, r, c, ai_color)
            if s >= 100000:
                return r, c

    # 2. 对手能形成活四或五连 → 堵
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            s = _evaluate_point(board, r, c, player_color)
            if s >= 100000:
                return r, c

    # 3. 自己能形成双活三（综合分≥20000）
    best_self = -1
    best_self_r = -1
    best_self_c = -1
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            s = _evaluate_point(board, r, c, ai_color)
            if s >= 20000 and s > best_self:
                best_self = s
                best_self_r = r
                best_self_c = c

    # 4. 对手能形成双活三 → 堵
    best_opp = -1
    best_opp_r = -1
    best_opp_c = -1
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            s = _evaluate_point(board, r, c, player_color)
            if s >= 20000 and s > best_opp:
                best_opp = s
                best_opp_r = r
                best_opp_c = c

    # 进攻优先于防守
    if best_self_r >= 0:
        return best_self_r, best_self_c
    if best_opp_r >= 0:
        return best_opp_r, best_opp_c

    return -1, -1


@njit(cache=True)
def _pick_top_candidates(board, moves_r, moves_c, count, ai_color, player_color, top_k):
    """从附近空位中选出综合评分最高的 top_k 个候选点"""
    scores = np.empty(count, dtype=np.float64)
    for i in range(count):
        r, c = moves_r[i], moves_c[i]
        atk = _evaluate_point(board, r, c, ai_color)
        dfn = _evaluate_point(board, r, c, player_color)
        scores[i] = atk * 1.5 + dfn

    # 选择排序取 top_k
    out_r = np.empty(top_k, dtype=np.int32)
    out_c = np.empty(top_k, dtype=np.int32)
    n = min(top_k, count)
    for i in range(n):
        best_idx = i
        best_score = scores[i]
        for j in range(i + 1, count):
            if scores[j] > best_score:
                best_score = scores[j]
                best_idx = j
        scores[i], scores[best_idx] = scores[best_idx], scores[i]
        out_r[i] = moves_r[best_idx]
        out_c[i] = moves_c[best_idx]
        moves_r[best_idx] = moves_r[i]
        moves_c[best_idx] = moves_c[i]
    return out_r, out_c, n


@njit(cache=True)
def _simulate(board, ai_color, player_color, cand_r, cand_c, cand_count, max_steps):
    """
    贪心模拟（Heavy Playout）：
    1. 每步先在候选集中检查必赢/必堵
    2. 无关键点则从候选集中选评分最高的（85%最优/15%次优）
    3. 最多走 max_steps 步，终止时评估候选集判断优劣
    """
    total = 0
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                total += 1
    current = 1 if total % 2 == 0 else 2
    opp = 3 - current

    for _ in range(min(max_steps, 225 - total)):
        # 1. 候选集中找自己必赢点
        for i in range(cand_count):
            r, c = cand_r[i], cand_c[i]
            if board[r, c] != 0:
                continue
            board[r, c] = current
            if _check_win(board, r, c, current):
                return current
            board[r, c] = 0

        # 2. 候选集中找对手必赢点（堵）
        block_r = -1
        block_c = -1
        for i in range(cand_count):
            r, c = cand_r[i], cand_c[i]
            if board[r, c] != 0:
                continue
            board[r, c] = opp
            if _check_win(board, r, c, opp):
                board[r, c] = 0
                block_r, block_c = r, c
                break
            board[r, c] = 0

        if block_r >= 0:
            board[block_r, block_c] = current
            current = opp
            opp = 3 - current
            continue

        # 3. 候选集中选评分最高的走法
        best_score = -1.0
        best_r = -1
        best_c = -1
        second_score = -1.0
        second_r = -1
        second_c = -1

        for i in range(cand_count):
            r, c = cand_r[i], cand_c[i]
            if board[r, c] != 0:
                continue
            atk = _evaluate_point(board, r, c, current)
            dfn = _evaluate_point(board, r, c, opp)
            s = atk * 1.5 + dfn
            if s > best_score:
                second_score = best_score
                second_r = best_r
                second_c = best_c
                best_score = s
                best_r = r
                best_c = c
            elif s > second_score:
                second_score = s
                second_r = r
                second_c = c

        if best_r >= 0:
            # 85% 最优，15% 次优
            if second_r >= 0 and np.random.random() < 0.15:
                r, c = second_r, second_c
            else:
                r, c = best_r, best_c
            board[r, c] = current
            if _check_win(board, r, c, current):
                return current
            current = opp
            opp = 1 - current
        else:
            break

    # 提前终止：评估候选集判断优劣
    ai_best = 0
    opp_best = 0
    for i in range(cand_count):
        r, c = cand_r[i], cand_c[i]
        if board[r, c] != 0:
            continue
        a = _evaluate_point(board, r, c, ai_color)
        o = _evaluate_point(board, r, c, player_color)
        if a > ai_best:
            ai_best = a
        if o > opp_best:
            opp_best = o

    if ai_best >= 100000:
        return ai_color
    elif opp_best >= 100000:
        return player_color
    elif ai_best > opp_best + 10000:
        return ai_color
    elif opp_best > ai_best + 10000:
        return player_color
    return 0


# ==================== Python 层包装函数 ====================

def get_nearby_moves(board):
    """返回附近空位列表 [(r,c), ...]"""
    nb = np.asarray(board, dtype=np.int8)
    moves_r, moves_c, count = _get_nearby_moves(nb)
    return [(int(moves_r[i]), int(moves_c[i])) for i in range(count)]


def get_candidate_moves(board, ai_color, player_color):
    """生成候选走法：综合进攻和防守评分，取 TOP_K"""
    nb = np.asarray(board, dtype=np.int8)
    moves_r, moves_c, count = _get_nearby_moves(nb)
    if count == 0:
        return []

    scored = []
    for i in range(count):
        r, c = int(moves_r[i]), int(moves_c[i])
        attack = _evaluate_point(nb, r, c, ai_color)
        defense = _evaluate_point(nb, r, c, player_color)
        score = attack * 1.5 + defense
        scored.append(((r, c), score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [move for move, _ in scored[:TOP_K]]


def get_forced_defense(board, player_color):
    """只有对手落子即赢时才强制堵"""
    nb = np.asarray(board, dtype=np.int8)
    r, c = _get_forced_defense(nb, player_color)
    if r >= 0:
        return (r, c)
    return None


def check_win(board, row, col, color):
    """胜负判定（兼容外部调用）"""
    nb = np.asarray(board, dtype=np.int8)
    return _check_win(nb, row, col, color)


# ==================== MCTS 节点 ====================
class Node:
    __slots__ = ('board', 'parent', 'move', 'color', 'children', 'visits', 'wins', 'candidates')

    def __init__(self, board, candidates, parent=None, move=None, color=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.color = color
        self.children = []
        self.visits = 0
        self.wins = 0
        self.candidates = candidates

    def is_fully_expanded(self):
        cand_r, cand_c, cand_count = self.candidates
        valid = 0
        for i in range(cand_count):
            if self.board[cand_r[i], cand_c[i]] == 0:
                valid += 1
        return len(self.children) == valid

    def get_untried_moves(self):
        tried = set(child.move for child in self.children)
        cand_r, cand_c, cand_count = self.candidates
        result = []
        for i in range(cand_count):
            r, c = int(cand_r[i]), int(cand_c[i])
            if self.board[r, c] == 0 and (r, c) not in tried:
                result.append((r, c))
        return result

    def best_child(self, c=UCB_C):
        return max(self.children, key=lambda child:
                   child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits))


# ==================== 主决策函数 ====================
def get_best_move(board, ai_color, player_color):
    nb = np.asarray(board, dtype=np.int8)
    empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if nb[r, c] == EMPTY]
    if not empty:
        return None

    # 1. 检查自己能否一步赢
    nb_copy = nb.copy()
    fr, fc = _get_forced_defense(nb_copy, ai_color)
    if fr >= 0:
        return (fr, fc)

    # 2. 只防"必赢点"
    forced = get_forced_defense(board, player_color)
    if forced is not None:
        return forced

    # 3. 战术检查：活四/双活三等明显威胁
    tr, tc = _get_tactical_move(nb, ai_color, player_color)
    if tr >= 0:
        return (tr, tc)

    # 4. 生成候选集
    cand_moves = get_candidate_moves(board, ai_color, player_color)
    if not cand_moves:
        return empty[0]

    cand_r = np.array([m[0] for m in cand_moves], dtype=np.int32)
    cand_c = np.array([m[1] for m in cand_moves], dtype=np.int32)
    cand_count = len(cand_moves)

    root = Node(board=nb.copy(), candidates=(cand_r, cand_c, cand_count))

    for _ in range(SIMULATIONS):
        node = root

        # Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()

        # Expansion
        untried = node.get_untried_moves()
        if untried:
            move = random.choice(untried)
            new_board = node.board.copy()

            total = int(np.count_nonzero(new_board))
            current_player = BLACK if total % 2 == 0 else WHITE

            new_board[move[0], move[1]] = current_player
            child = Node(board=new_board, candidates=(cand_r, cand_c, cand_count),
                         parent=node, move=move, color=current_player)
            node.children.append(child)
            node = child

        # Simulation
        sim_board = node.board.copy()
        winner = _simulate(sim_board, ai_color, player_color, cand_r, cand_c, cand_count, SIM_MAX_STEPS)

        # Backpropagation
        while node is not None:
            node.visits += 1
            if node.color == winner:
                node.wins += 1
            node = node.parent

    if not root.children:
        return empty[0]
    best_child_node = max(root.children, key=lambda child: child.visits)
    return best_child_node.move
