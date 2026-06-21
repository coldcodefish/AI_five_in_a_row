# AI/ai_mcts.py
import random
import math
import copy
from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE, evaluate_point

# MCTS 参数
SIMULATIONS = 100  # 每次决策的模拟次数（可调）
UCB_C = 1.4        # 探索常数

class Node:
    def __init__(self, board, parent=None, move=None, color=None):
        self.board = board          # 当前棋盘状态（深拷贝）
        self.parent = parent        # 父节点
        self.move = move            # 导致此节点的走法 (row, col)
        self.color = color          # 在此节点落子的颜色（用于回溯）
        self.children = []          # 子节点列表
        self.visits = 0             # 访问次数
        self.wins = 0               # 模拟胜利次数（从AI视角）

    def is_fully_expanded(self):
        # 如果所有空位都已扩展，则完全展开
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == EMPTY]
        return len(self.children) == len(empty)

    def get_untried_moves(self):
        # 获取尚未尝试的走法
        tried = [child.move for child in self.children]
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r][c] == EMPTY]
        return [move for move in empty if move not in tried]

    def best_child(self, c=UCB_C):
        # 使用 UCB 公式选择最佳子节点
        return max(self.children, key=lambda child: child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits))

def get_best_move(board, ai_color, player_color):
    root = Node(board=copy.deepcopy(board), parent=None, move=None, color=None)
    for _ in range(SIMULATIONS):
        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # 2. Expansion
        untried = node.get_untried_moves()
        if untried:
            move = random.choice(untried)
            new_board = copy.deepcopy(node.board)
            # 确定落子颜色：轮到谁走？根据棋步奇偶性或当前局面判断
            # 这里简化：通过计算已落子数量决定
            total_stones = sum(cell != EMPTY for row in node.board for cell in row)
            current_player = BLACK if total_stones % 2 == 0 else WHITE
            color = current_player   # 然后在 backprop 中根据 color == winner 等逻辑
            new_board[move[0]][move[1]] = color
            child = Node(board=new_board, parent=node, move=move, color=color)
            node.children.append(child)
            node = child
        # 3. Simulation (随机对局直到终局)
        winner = simulate(node.board, ai_color, player_color)
        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            if node.color == ai_color and winner == ai_color:
                node.wins += 1
            elif node.color == player_color and winner == player_color:
                # 如果节点颜色是玩家且玩家赢了，从AI视角看是失败，不加分
                pass
            # 实际上，我们更关心 AI 的胜利，所以可以从 AI 角度计分
            # 简便方法：如果 winner == ai_color，所有祖先节点 wins++？
            # 更好的做法：从根到叶，如果最终获胜者是该节点落子方，则加分
            # 我们采用常见做法：更新时根据节点落子颜色是否等于获胜者加分
            if node.color == winner:
                node.wins += 1
            node = node.parent

    # 选择访问次数最多的子节点作为最佳走法
    if not root.children:
        return None
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move

def simulate(board, ai_color, player_color):
    """随机模拟对局直到终局，返回获胜者（BLACK/WHITE/None平局）"""
    board_sim = copy.deepcopy(board)
    current = ai_color  # AI先手（或根据实际情况）
    # 最多模拟 200 步防止死循环
    for _ in range(200):
        empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board_sim[r][c] == EMPTY]
        if not empty:
            return None  # 平局
        move = random.choice(empty)
        board_sim[move[0]][move[1]] = current
        # 检查胜利
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