# AI/ai_dqn_train.py
"""
DQN 五子棋 AI 训练脚本（15×15 棋盘）
训练完成后会在当前目录生成 dqn_model.ckpt
"""

import os
import random
import numpy as np
from collections import deque
from .ai_rating import BOARD_SIZE, EMPTY, BLACK, WHITE

import tensorflow as tf
if tf.__version__.startswith('2'):
    tf = tf.compat.v1
    tf.compat.v1.disable_v2_behavior()

# ---------- 超参数 ----------
L1_NUM = BOARD_SIZE * BOARD_SIZE * 2   # 450
L2_NUM = 450
L3_NUM = 256
L4_NUM = 128
N_ACTIONS = BOARD_SIZE * BOARD_SIZE    # 225

LEARNING_RATE = 0.001
REWARD_DECAY = 0.9
EPSILON_MAX = 0.9
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
REPLACE_TARGET_ITER = 200
MEMORY_SIZE = 5000
BATCH_SIZE = 128
TRAIN_EPISODES = 50000
SAVE_PATH = 'dqn_model.ckpt'

# ---------- DQN 智能体 ----------
class DQNAgent:
    def __init__(self):
        self.learn_step_counter = 0
        self.epsilon = EPSILON_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, L1_NUM], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, N_ACTIONS], name='q_target')
        w_init = tf.random_normal_initializer(0., 0.3)
        b_init = tf.constant_initializer(0.1)

        with tf.variable_scope('eval_net'):
            self.q_eval = self._build_layers(self.s, L1_NUM, L2_NUM, L3_NUM, L4_NUM, N_ACTIONS,
                                             w_init, b_init,
                                             collections=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES])
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

        self.s_ = tf.placeholder(tf.float32, [None, L1_NUM], name='s_')
        with tf.variable_scope('target_net'):
            self.q_next = self._build_layers(self.s_, L1_NUM, L2_NUM, L3_NUM, L4_NUM, N_ACTIONS,
                                             w_init, b_init,
                                             collections=['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES])

        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def _build_layers(self, inputs, n1, n2, n3, n4, n_actions, w_init, b_init, collections):
        with tf.variable_scope('l1'):
            w1 = tf.get_variable('w1', [n1, n1], initializer=w_init, collections=collections)
            b1 = tf.get_variable('b1', [1, n1], initializer=b_init, collections=collections)
            l1 = tf.nn.relu(tf.matmul(inputs, w1) + b1)
        with tf.variable_scope('l2'):
            w2 = tf.get_variable('w2', [n1, n2], initializer=w_init, collections=collections)
            b2 = tf.get_variable('b2', [1, n2], initializer=b_init, collections=collections)
            l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        with tf.variable_scope('l3'):
            w3 = tf.get_variable('w3', [n2, n3], initializer=w_init, collections=collections)
            b3 = tf.get_variable('b3', [1, n3], initializer=b_init, collections=collections)
            l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)
        with tf.variable_scope('l4'):
            w4 = tf.get_variable('w4', [n3, n4], initializer=w_init, collections=collections)
            b4 = tf.get_variable('b4', [1, n4], initializer=b_init, collections=collections)
            l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
        with tf.variable_scope('l5'):
            w5 = tf.get_variable('w5', [n4, n_actions], initializer=w_init, collections=collections)
            b5 = tf.get_variable('b5', [1, n_actions], initializer=b_init, collections=collections)
            return tf.matmul(l4, w5) + b5

    def choose_action(self, board, ai_color):
        state = self._board_to_input(board, ai_color)
        if np.random.rand() < self.epsilon:
            empty = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c] == EMPTY]
            return random.choice(empty) if empty else None
        q_values = self.sess.run(self.q_eval, feed_dict={self.s: [state]})[0]
        best_val = -np.inf
        best_move = None
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == EMPTY:
                    idx = r * BOARD_SIZE + c
                    if q_values[idx] > best_val:
                        best_val = q_values[idx]
                        best_move = (r, c)
        return best_move

    def _board_to_input(self, board, ai_color):
        opp_color = WHITE if ai_color == BLACK else BLACK
        ch0 = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        ch1 = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if board[r][c] == ai_color:
                    ch0[r][c] = 1.0
                elif board[r][c] == opp_color:
                    ch1[r][c] = 1.0
        return np.hstack([ch0.flatten(), ch1.flatten()])

    def store_transition(self, s, a, r, s_):
        self.memory.append((s, a, r, s_))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
        if self.learn_step_counter % REPLACE_TARGET_ITER == 0:
            self.sess.run(self.replace_target_op)

        batch = random.sample(self.memory, BATCH_SIZE)
        s = np.array([t[0] for t in batch])
        a = np.array([t[1] for t in batch])
        r = np.array([t[2] for t in batch])
        s_ = np.array([t[3] for t in batch])

        q_next, q_eval = self.sess.run([self.q_next, self.q_eval],
                                       feed_dict={self.s_: s_, self.s: s})
        q_target = q_eval.copy()
        for i in range(BATCH_SIZE):
            a_idx = a[i][0] * BOARD_SIZE + a[i][1]
            q_target[i, a_idx] = r[i] + REWARD_DECAY * np.max(q_next[i])

        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.s: s, self.q_target: q_target})
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
        self.learn_step_counter += 1

    def save_model(self, path):
        self.saver.save(self.sess, path)

# ---------- 辅助函数 ----------
def check_win(board):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = board[r][c]
            if color == EMPTY:
                continue
            for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
                cnt = 1
                nr, nc = r+dr, c+dc
                while 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==color:
                    cnt += 1; nr+=dr; nc+=dc
                nr, nc = r-dr, c-dc
                while 0<=nr<BOARD_SIZE and 0<=nc<BOARD_SIZE and board[nr][nc]==color:
                    cnt += 1; nr-=dr; nc-=dc
                if cnt >= 5:
                    return color
    return None

# ---------- 修正后的 simulate_game ----------
def simulate_game(agent, ai_color):
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    current_player = BLACK
    experiences = []  # 元素为 (s, a, r, s_)
    final_r = None     # 结局奖励

    while True:
        # 当前玩家走子
        if current_player == ai_color:
            move = agent.choose_action(board, ai_color)
            if move is None:  # AI无子可走（棋盘已满）
                final_r = 0.0
                break
            r, c = move
            s = agent._board_to_input(board, ai_color)
            board[r][c] = current_player
            experiences.append((s, move, 0, None))  # 奖励和s_待填充
        else:
            empty = [(r,c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if board[r][c]==EMPTY]
            if not empty:   # 对手无子可走（理论上不应发生，但保护）
                final_r = 0.0
                break
            move = random.choice(empty)
            board[move[0]][move[1]] = current_player

        # 检查胜负
        winner = check_win(board)
        if winner is not None:
            final_r = 1.0 if winner == ai_color else -1.0
            break

        # 检查平局（棋盘满）
        if all(board[r][c]!=EMPTY for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)):
            final_r = 0.0
            break

        current_player = WHITE if current_player == BLACK else BLACK

    # 如果 final_r 仍未赋值（理论上不会），设为平局
    if final_r is None:
        final_r = 0.0

    # 填充所有经验的 s_ 和奖励
    for i in range(len(experiences)):
        s, a, _, _ = experiences[i]
        if i == len(experiences) - 1:
            s_ = agent._board_to_input(board, ai_color)
        else:
            s_ = experiences[i+1][0]
        experiences[i] = (s, a, final_r, s_)

    return experiences

# ---------- 训练主循环 ----------
def train():
    agent = DQNAgent()
    total_steps = 0

    for episode in range(1, TRAIN_EPISODES+1):
        ai_color = BLACK if np.random.rand() < 0.5 else WHITE
        exps = simulate_game(agent, ai_color)
        for s, a, r, s_ in exps:
            agent.store_transition(s, a, r, s_)
            agent.learn()
            total_steps += 1

        if episode % 1000 == 0:
            print(f'Episode {episode}, Steps {total_steps}, Epsilon {agent.epsilon:.3f}')

        if episode % 5000 == 0:
            agent.save_model(SAVE_PATH)
            print('Model saved.')

    agent.save_model(SAVE_PATH)
    print('Training finished. Model saved as', SAVE_PATH)

if __name__ == '__main__':
    train()