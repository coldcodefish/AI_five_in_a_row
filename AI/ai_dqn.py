"""
DQN 五子棋 AI（PyTorch 实现，15×15 棋盘）
推理接口：get_best_move(board, ai_color, player_color) -> (row, col)
模型文件：dqn_model.pt
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE

BOARD = BOARD_SIZE
N_ACTIONS = BOARD * BOARD

class DQNNet(nn.Module):
    """轻量 CNN：输入 (B, 2, 15, 15)，输出 (B, 225) Q 值"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * BOARD * BOARD, 256)
        self.fc2 = nn.Linear(256, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


_global_net = None
_global_device = None


def _get_net():
    global _global_net, _global_device
    if _global_net is None:
        _global_device = torch.device('cpu')
        _global_net = DQNNet().to(_global_device)
        _global_net.eval()

        model_path = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=_global_device, weights_only=True)
            _global_net.load_state_dict(state)
            print(f'[DQN] 模型加载成功: {model_path}')
        else:
            print(f'[DQN] 未找到模型文件 {model_path}，使用随机权重')
    return _global_net, _global_device


def _board_to_tensor(board, ai_color):
    """将棋盘转换为网络输入张量 (1, 2, 15, 15)：通道0=己方，通道1=对手"""
    opp_color = WHITE if ai_color == BLACK else BLACK
    arr = np.array(board, dtype=np.float32)
    ch_self = (arr == ai_color).astype(np.float32)
    ch_opp = (arr == opp_color).astype(np.float32)
    tensor = np.stack([ch_self, ch_opp], axis=0)
    return torch.from_numpy(tensor).unsqueeze(0)


def _get_legal_mask(board):
    """返回合法动作掩码 (225,)，True 表示可下"""
    arr = np.array(board).flatten()
    return arr == EMPTY


def _get_candidate_moves(board, radius=1):
    """返回候选落子位置（已有棋子周围 radius 范围内的空位），空盘返回中心点"""
    arr = np.array(board)
    candidates = set()
    has_stone = False
    for r in range(BOARD):
        for c in range(BOARD):
            if arr[r, c] != EMPTY:
                has_stone = True
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD and 0 <= nc < BOARD and arr[nr, nc] == EMPTY:
                            candidates.add((nr, nc))
    if not has_stone:
        return [(BOARD // 2, BOARD // 2)]
    return list(candidates)


def _check_win_at(board, row, col, color):
    """检查在 (row, col) 落 color 子后是否五连"""
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        cnt = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == color:
            cnt += 1; r += dr; c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == color:
            cnt += 1; r -= dr; c -= dc
        if cnt >= 5:
            return True
    return False


def _find_critical_move(board, color, candidates):
    """在候选位置中找必胜落子（自己一步赢）"""
    for r, c in candidates:
        board[r][c] = color
        if _check_win_at(board, r, c, color):
            board[r][c] = EMPTY
            return (r, c)
        board[r][c] = EMPTY
    return None


def get_best_move(board, ai_color, player_color=None):
    """
    DQN AI 落子接口
    :return: (row, col) 坐标
    """
    candidates = _get_candidate_moves(board, radius=1)
    if not candidates:
        for r in range(BOARD):
            for c in range(BOARD):
                if board[r][c] == EMPTY:
                    return r, c
        return 0, 0

    opp_color = WHITE if ai_color == BLACK else BLACK

    # 战术规则：自己一步赢 → 立即赢；对手一步赢 → 必须堵
    win_move = _find_critical_move(board, ai_color, candidates)
    if win_move is not None:
        return win_move

    block_move = _find_critical_move(board, opp_color, candidates)
    if block_move is not None:
        return block_move

    # 神经网络决策
    net, device = _get_net()
    state = _board_to_tensor(board, ai_color).to(device)
    with torch.no_grad():
        q_values = net(state).cpu().numpy()[0]

    best_move = None
    best_q = -float('inf')
    for r, c in candidates:
        idx = r * BOARD + c
        if q_values[idx] > best_q:
            best_q = q_values[idx]
            best_move = (r, c)

    return best_move
