import tensorflow.compat.v1 as tf
import numpy as np
import os

tf.disable_v2_behavior()

# ---------- 超参数（与您原代码保持一致） ----------
l_1num = 722
l_2num = 722
l_3num = 540
l_4num = 361
col = 19
n_actions = 361

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.18,
            reward_decay=0.01,
            e_greedy=0.9,
            replace_target_iter=50,
            memory_size=500,
            batch_size=100,
            e_greedy_increment=0.0001,
            output_graph=False,
            savefile='varriable.ckpt'
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 1.0
        self.n_r = 1
        self.learn_step_counter = 0
        self.memory_counter = 0

        # 经验池（仅用于占位，推理时不需要）
        self.memory = np.zeros((self.memory_size, l_1num * 2 + 2))

        self._build_net()
        self.sess = tf.Session()
        self.savefile = savefile
        self.saver = tf.train.Saver()
        self.path = os.path.abspath('.') + '/meta/' + self.savefile
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ---------- eval 网络 ----------
        self.s = tf.placeholder(tf.float32, [None, l_1num], name='s')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [l_1num, l_1num], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, l_1num], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [l_1num, l_2num], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, l_2num], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [l_2num, l_3num], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, l_3num], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [l_3num, l_4num], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, l_4num], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [l_4num, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l4, w5) + b5

        # 损失和优化器（推理时不会用到，但保留定义）
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ---------- target 网络（推理时不需要，但保留结构） ----------
        self.s_ = tf.placeholder(tf.float32, [None, l_1num], name='s_')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [l_1num, l_1num], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, l_1num], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [l_1num, l_2num], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, l_2num], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [l_2num, l_3num], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, l_3num], initializer=b_initializer, collections=c_names)
                l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

            with tf.variable_scope('l4'):
                w4 = tf.get_variable('w4', [l_3num, l_4num], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable('b4', [1, l_4num], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable('w5', [l_4num, self.n_actions], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable('b5', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l4, w5) + b5

    def choose_action(self, qipan, observation):
        """
        返回动作索引 (0~360)，使用最大Q值（epsilon=0，即贪婪策略）
        """
        ob = np.reshape(observation, [1, l_1num])
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: ob})[0]
        action_list = np.reshape(qipan, [1, self.n_features])

        # 收集合法动作（棋子周围8方向）
        legal = set()
        for i in range(self.n_actions):
            if action_list[0, i] == 1 or action_list[0, i] == 2:
                neighbors = [
                    i - col, i - 1, i + 1, i + col,
                    i - col - 1, i - col + 1,
                    i + col - 1, i + col + 1
                ]
                for ni in neighbors:
                    if 0 <= ni < self.n_actions and action_list[0, ni] == 0:
                        legal.add(ni)
        if not legal:
            empty = [idx for idx in range(self.n_actions) if action_list[0, idx] == 0]
            if empty:
                return np.random.choice(empty)
            else:
                return np.random.randint(0, self.n_actions)

        # 在合法动作中选择Q值最大的（epsilon=0，完全贪婪）
        action = max(legal, key=lambda a: actions_value[a])
        return action

    def getvarriable(self):
        if os.path.exists(self.path + '.meta') or os.path.exists(self.path):
            self.saver.restore(self.sess, self.path)
            print('模型加载成功')
        else:
            print('未找到预训练模型，从零开始')


# ---------- 全局单例 ----------
_global_dqn = None


def _get_dqn():
    global _global_dqn
    if _global_dqn is None:
        _global_dqn = DeepQNetwork(n_actions=361, n_features=361)
        _global_dqn.getvarriable()   # 尝试加载权重
        # 强制使用贪婪策略（epsilon=0），确保推理确定性
        _global_dqn.epsilon = 0.0
    return _global_dqn


# ---------- 对外接口 ----------
def get_best_move(board, ai_color=None, player_color=None):
    """
    DQN AI 落子接口，供其他模块调用
    :param board: 棋盘，支持 19x19 的列表、元组、numpy数组，或一维数组(361,)
    :param ai_color: 占位，忽略（DQN不依赖颜色）
    :param player_color: 占位，忽略
    :return: (row, col) 坐标，如 (5, 7)
    """
    dqn = _get_dqn()

    # 1. 转换为 numpy 数组
    if not isinstance(board, np.ndarray):
        board = np.array(board)

    # 2. 处理形状：支持 (19,19), (361,), (1,361) 等
    if board.ndim == 1:
        if board.shape[0] == 361:
            board = board.reshape(19, 19)
        else:
            raise ValueError(f"一维棋盘长度必须为361，实际为 {board.shape[0]}")
    elif board.ndim == 2:
        if board.shape == (1, 361):
            board = board.reshape(19, 19)
        elif board.shape != (19, 19):
            raise ValueError(f"棋盘应为 19x19 矩阵，实际为 {board.shape}")
    else:
        raise ValueError("棋盘维度应为 1 或 2")

    # 3. 确保数据类型为 int
    board = board.astype(np.int8)

    # 4. 构建 observation (722 维)：白子(1)和黑子(2)的one-hot
    flat = board.flatten()
    w_qipan = (flat == 1).astype(int)   # 假设白子=1
    b_qipan = (flat == 2).astype(int)   # 假设黑子=2
    observation = np.hstack((w_qipan, b_qipan))  # (722,)

    # 5. 获取动作索引
    action = dqn.choose_action(board, observation)

    # 6. 转换为坐标
    row = action // 19
    col = action % 19
    return row, col