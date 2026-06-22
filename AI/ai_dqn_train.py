"""
DQN 五子棋 AI 训练脚本（PyTorch 实现，15×15 棋盘）
两阶段训练（AlphaGo 式）：
  阶段1 - 模仿学习预训练：学习 negamax 的落子策略
  阶段2 - DQN 强化学习微调：在预训练基础上自我对弈优化
训练完成后生成 dqn_model.pt
"""
import os
import time
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ai_dqn import DQNNet, BOARD, N_ACTIONS
from .ai_rating import (
    BOARD_SIZE, EMPTY, BLACK, WHITE,
    get_best_move as rating_get_best_move,
)
from . import ai_negamax

SAVE_PATH = os.path.join(os.path.dirname(__file__), 'dqn_model.pt')

# 模仿学习超参数
IMIT_GAMES = 200
IMIT_EPOCHS = 30
IMIT_LR = 1e-3
IMIT_BATCH = 128
NEGAMAX_TIME = 0.25
SKIP_DQN = True               # 跳过 DQN 微调（实测会退化模型）

# DQN 微调超参数
DQN_EPISODES = 2500
DQN_LR = 5e-5
GAMMA = 0.90
EPSILON_START = 0.2
EPSILON_END = 0.05
EPSILON_DECAY = 0.999
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 10000
BATCH_SIZE = 64
LEARN_EVERY = 4
SAVE_EVERY = 300
EVAL_EVERY = 300

# 奖励权重
R_WIN = 1.0
R_LOSE = -1.0
R_DRAW = 0.0
R_LIVE_FOUR = 0.3
R_LIVE_THREE = 0.08
R_RUSH_FOUR = 0.1


def check_win(board, row, col):
    color = board[row][col]
    if color == EMPTY:
        return False
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


def count_line_at(board, row, col, color):
    results = []
    for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        cnt = 1; open_ends = 0
        r, c = row + dr, col + dc
        while 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == color:
            cnt += 1; r += dr; c += dc
        if 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == EMPTY:
            open_ends += 1
        r, c = row - dr, col - dc
        while 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == color:
            cnt += 1; r -= dr; c -= dc
        if 0 <= r < BOARD and 0 <= c < BOARD and board[r][c] == EMPTY:
            open_ends += 1
        results.append((cnt, open_ends))
    return results


def shape_reward(board, row, col, color):
    reward = 0.0
    for cnt, open_ends in count_line_at(board, row, col, color):
        if cnt >= 5:
            continue
        if cnt == 4:
            if open_ends >= 2:
                reward += R_LIVE_FOUR
            elif open_ends == 1:
                reward += R_RUSH_FOUR
        elif cnt == 3:
            if open_ends >= 2:
                reward += R_LIVE_THREE
    return reward


def board_full(board):
    for r in range(BOARD):
        for c in range(BOARD):
            if board[r][c] == EMPTY:
                return False
    return True


def get_candidate_moves(board, radius=1):
    candidates = set()
    has_stone = False
    for r in range(BOARD):
        for c in range(BOARD):
            if board[r][c] != EMPTY:
                has_stone = True
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < BOARD and 0 <= nc < BOARD and board[nr][nc] == EMPTY:
                            candidates.add((nr, nc))
    if not has_stone:
        return [(BOARD // 2, BOARD // 2)]
    return list(candidates)


def board_to_tensor(board, color):
    opp = WHITE if color == BLACK else BLACK
    arr = np.array(board, dtype=np.float32)
    ch_self = (arr == color).astype(np.float32)
    ch_opp = (arr == opp).astype(np.float32)
    tensor = np.stack([ch_self, ch_opp], axis=0)
    return torch.from_numpy(tensor).unsqueeze(0)


# ================================================================
# 阶段1：模仿学习预训练
# ================================================================
def generate_imitation_data(n_games):
    """用 negamax 自我对弈生成 (state, action) 训练数据"""
    original_time = ai_negamax.TIME_LIMIT
    ai_negamax.TIME_LIMIT = NEGAMAX_TIME

    states = []
    actions = []
    t0 = time.time()
    try:
        for g in range(n_games):
            board = [[EMPTY] * BOARD for _ in range(BOARD)]
            current = BLACK
            for step in range(225):
                opp = WHITE if current == BLACK else BLACK
                move = ai_negamax.get_best_move(board, current, opp)
                if move is None:
                    break
                states.append(board_to_tensor(board, current))
                actions.append(move[0] * BOARD + move[1])

                r, c = move
                board[r][c] = current
                if check_win(board, r, c) or board_full(board):
                    break
                current = WHITE if current == BLACK else BLACK
            if (g + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f'  生成数据: {g+1}/{n_games} 局, 样本数={len(states)}, '
                      f'用时={elapsed:.0f}s, {elapsed/(g+1):.1f}s/局')
    finally:
        ai_negamax.TIME_LIMIT = original_time
    return states, actions


def imitation_pretrain(net, states, actions, device=torch.device('cpu')):
    """用交叉熵损失训练网络模仿 negamax"""
    optimizer = torch.optim.Adam(net.parameters(), lr=IMIT_LR)
    n = len(states)
    print(f'  样本数: {n}, 开始训练 {IMIT_EPOCHS} 轮')

    for epoch in range(IMIT_EPOCHS):
        idx = list(range(n))
        random.shuffle(idx)
        total_loss = 0.0
        correct = 0
        batches = 0
        for i in range(0, n, IMIT_BATCH):
            batch_idx = idx[i:i + IMIT_BATCH]
            s = torch.cat([states[j] for j in batch_idx]).to(device)
            a = torch.tensor([actions[j] for j in batch_idx], dtype=torch.long, device=device)

            logits = net(s)
            loss = F.cross_entropy(logits, a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == a).sum().item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)
        acc = correct / n
        print(f'  Epoch {epoch+1}/{IMIT_EPOCHS} | loss={avg_loss:.4f} | acc={acc:.3f}')
    return net


# ================================================================
# 阶段2：DQN 强化学习微调
# ================================================================
class DQNAgent:
    def __init__(self, eval_net=None, device=torch.device('cpu')):
        self.device = device
        self.eval_net = eval_net if eval_net is not None else DQNNet().to(device)
        self.target_net = DQNNet().to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=DQN_LR)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.learn_step = 0
        self.epsilon = EPSILON_START

    def choose_action(self, board, ai_color, greedy=False):
        candidates = get_candidate_moves(board, radius=1)
        if not candidates:
            return None
        if not greedy and random.random() < self.epsilon:
            return random.choice(candidates)
        state = board_to_tensor(board, ai_color).to(self.device)
        with torch.no_grad():
            q = self.eval_net(state).cpu().numpy()[0]
        best_move = None
        best_q = -float('inf')
        for r, c in candidates:
            idx = r * BOARD + c
            if q[idx] > best_q:
                best_q = q[idx]
                best_move = (r, c)
        return best_move

    def store(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        if self.learn_step % REPLACE_TARGET_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        batch = random.sample(self.memory, BATCH_SIZE)
        s, a, r, s_, done = zip(*batch)
        s = torch.cat(s).to(self.device)
        s_ = torch.cat(s_).to(self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        q_eval = self.eval_net(s)
        q_eval_a = q_eval.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN：用 eval_net 选动作，target_net 估值
        with torch.no_grad():
            q_next_eval = self.eval_net(s_)
            q_next_target = self.target_net(s_)
            max_a = q_next_eval.argmax(dim=1)
            q_next = q_next_target.gather(1, max_a.unsqueeze(1)).squeeze(1)
            q_target = r + GAMMA * q_next * (1 - done)

        loss = F.mse_loss(q_eval_a, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 1.0)
        self.optimizer.step()
        self.learn_step += 1

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)


def play_one_game(agent, ai_color, opp_rating_ratio):
    """AI 与对手对弈一局"""
    board = [[EMPTY] * BOARD for _ in range(BOARD)]
    current = BLACK
    experiences = []
    final_reward = None
    winner = None
    use_rating = random.random() < opp_rating_ratio
    opp_color = WHITE if ai_color == BLACK else BLACK

    while True:
        if current == ai_color:
            s = board_to_tensor(board, ai_color)
            move = agent.choose_action(board, ai_color)
            if move is None:
                final_reward = R_DRAW
                break
            r, c = move
            board[r][c] = ai_color
            shaped = shape_reward(board, r, c, ai_color)
            if check_win(board, r, c):
                final_reward = R_WIN + shaped
                winner = ai_color
                s_ = board_to_tensor(board, ai_color)
                experiences.append((s, r * BOARD + c, final_reward, s_, True))
                break
            experiences.append((s, r * BOARD + c, shaped, None, False))
        else:
            if use_rating:
                move = rating_get_best_move(board, opp_color, ai_color)
            else:
                cands = get_candidate_moves(board, radius=1)
                move = random.choice(cands) if cands else None
            if move is None:
                final_reward = R_DRAW
                break
            r, c = move
            board[r][c] = opp_color
            if check_win(board, r, c):
                final_reward = R_LOSE
                winner = opp_color
                break
            if board_full(board):
                final_reward = R_DRAW
                break
        current = WHITE if current == BLACK else BLACK

    if final_reward is None:
        final_reward = R_DRAW

    for i in range(len(experiences)):
        s, a, r, s_, done = experiences[i]
        if i == len(experiences) - 1:
            if s_ is None:
                s_ = board_to_tensor(board, ai_color)
            experiences[i] = (s, a, final_reward, s_, True)
        else:
            if s_ is None:
                s_ = experiences[i + 1][0]
            experiences[i] = (s, a, r, s_, False)
    return experiences, winner


def evaluate(agent, n_games=10):
    wins = 0
    for _ in range(n_games):
        ai_color = BLACK if random.random() < 0.5 else WHITE
        board = [[EMPTY] * BOARD for _ in range(BOARD)]
        current = BLACK
        steps = 0
        while steps < 225:
            if current == ai_color:
                move = agent.choose_action(board, ai_color, greedy=True)
            else:
                opp_color = WHITE if ai_color == BLACK else BLACK
                move = rating_get_best_move(board, opp_color, ai_color)
            if move is None:
                break
            r, c = move
            board[r][c] = current
            if check_win(board, r, c):
                if current == ai_color:
                    wins += 1
                break
            if board_full(board):
                break
            current = WHITE if current == BLACK else BLACK
            steps += 1
    return wins / n_games


def evaluate_random(net, n_games=20):
    """评估对随机对手的胜率"""
    wins = 0
    for _ in range(n_games):
        ai_color = BLACK if random.random() < 0.5 else WHITE
        board = [[EMPTY] * BOARD for _ in range(BOARD)]
        current = BLACK
        steps = 0
        while steps < 225:
            if current == ai_color:
                state = board_to_tensor(board, ai_color)
                with torch.no_grad():
                    q = net(state).cpu().numpy()[0]
                cands = get_candidate_moves(board, radius=1)
                if not cands:
                    break
                move = max(cands, key=lambda x: q[x[0] * BOARD + x[1]])
            else:
                cands = get_candidate_moves(board, radius=1)
                if not cands:
                    break
                move = random.choice(cands)
            r, c = move
            board[r][c] = current
            if check_win(board, r, c):
                if current == ai_color:
                    wins += 1
                break
            if board_full(board):
                break
            current = WHITE if current == BLACK else BLACK
            steps += 1
    return wins / n_games


def evaluate_simple(net, n_games=10):
    """评估网络对评分AI的胜率（直接用网络推理，不经 agent）"""
    wins = 0
    for _ in range(n_games):
        ai_color = BLACK if random.random() < 0.5 else WHITE
        board = [[EMPTY] * BOARD for _ in range(BOARD)]
        current = BLACK
        steps = 0
        while steps < 225:
            if current == ai_color:
                state = board_to_tensor(board, ai_color)
                with torch.no_grad():
                    q = net(state).cpu().numpy()[0]
                cands = get_candidate_moves(board, radius=1)
                if not cands:
                    break
                move = max(cands, key=lambda x: q[x[0] * BOARD + x[1]])
            else:
                opp_color = WHITE if ai_color == BLACK else BLACK
                move = rating_get_best_move(board, opp_color, ai_color)
                if move is None:
                    break
            r, c = move
            board[r][c] = current
            if check_win(board, r, c):
                if current == ai_color:
                    wins += 1
                break
            if board_full(board):
                break
            current = WHITE if current == BLACK else BLACK
            steps += 1
    return wins / n_games


# ================================================================
# 主训练函数
# ================================================================
def train():
    device = torch.device('cpu')
    print('=' * 60)
    print('DQN 五子棋训练（两阶段：模仿学习 + DQN微调）')
    print(f'棋盘: {BOARD}x{BOARD}  动作数: {N_ACTIONS}')
    print('=' * 60)

    total_start = time.time()

    # ---------- 阶段1：模仿学习 ----------
    print('\n[阶段1] 模仿学习预训练')
    print('-' * 40)
    t0 = time.time()
    print('生成训练数据（negamax 自我对弈）...')
    states, actions = generate_imitation_data(IMIT_GAMES)
    print(f'数据生成完成: {len(states)} 样本, 用时 {time.time()-t0:.0f}s')

    net = DQNNet().to(device)
    t0 = time.time()
    net = imitation_pretrain(net, states, actions, device)
    print(f'模仿训练完成, 用时 {time.time()-t0:.0f}s')

    torch.save(net.state_dict(), SAVE_PATH)
    print(f'预训练模型已保存: {SAVE_PATH}')

    net.eval()
    wr = evaluate_random(net, 20)
    print(f'预训练后对随机胜率: {wr*100:.0f}%')

    if SKIP_DQN:
        print('\n[跳过 DQN 微调] 仅使用模仿学习模型（DQN 微调实测会退化）')
        print(f'\n训练完成！模型已保存: {SAVE_PATH}')
        print(f'总用时: {time.time() - total_start:.0f}s')
        wr_rand = evaluate_random(net, 30)
        wr_rating = evaluate_simple(net, 10)
        print(f'最终评估: 对随机胜率={wr_rand*100:.0f}%  对评分AI胜率={wr_rating*100:.0f}%')
        return

    # ---------- 阶段2：DQN 微调 ----------
    print('\n[阶段2] DQN 强化学习微调')
    print('-' * 40)
    net.train()
    agent = DQNAgent(eval_net=net, device=device)
    start_time = time.time()
    total_steps = 0

    for episode in range(1, DQN_EPISODES + 1):
        progress = episode / DQN_EPISODES
        opp_ratio = 0.5 + 0.3 * progress   # 0.5 -> 0.8
        ai_color = BLACK if random.random() < 0.5 else WHITE
        exps, winner = play_one_game(agent, ai_color, opp_ratio)

        for s, a, r, s_, done in exps:
            agent.store(s, a, r, s_, done)
            total_steps += 1
            if total_steps % LEARN_EVERY == 0:
                agent.learn()

        agent.decay_epsilon()

        if episode % 50 == 0:
            elapsed = time.time() - start_time
            speed = episode / elapsed if elapsed > 0 else 0
            print(f'Episode {episode:5d} | eps={agent.epsilon:.3f} | '
                  f'steps={total_steps} | {speed:.1f} ep/s | '
                  f'elapsed={elapsed:.0f}s | winner={winner}')

        if episode % EVAL_EVERY == 0:
            agent.eval_net.eval()
            wr_rand = evaluate_random(agent.eval_net, 20)
            wr_rating = evaluate(agent, 10)
            agent.eval_net.train()
            print(f'  >> 对随机胜率={wr_rand*100:.0f}%  对评分AI胜率={wr_rating*100:.0f}%')

        if episode % SAVE_EVERY == 0:
            agent.save(SAVE_PATH)
            print(f'  >> 模型已保存: {SAVE_PATH}')

        # 时间限制：总训练不超过 80 分钟
        if time.time() - total_start > 4800:
            print('达到时间上限，提前停止')
            break

    agent.save(SAVE_PATH)
    print(f'\n训练完成！模型已保存: {SAVE_PATH}')
    print(f'总用时: {time.time() - total_start:.0f}s')

    agent.eval_net.eval()
    wr_rand = evaluate_random(agent.eval_net, 30)
    wr_rating = evaluate(agent, 10)
    print(f'最终评估: 对随机胜率={wr_rand*100:.0f}%  对评分AI胜率={wr_rating*100:.0f}%')


if __name__ == '__main__':
    train()
