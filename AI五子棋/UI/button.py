# UI/button.py
# 导入 PySide6 中的基础控件类
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QSizePolicy)
# 导入 Qt 核心常量、定时器、信号类
from PySide6.QtCore import Qt, QTimer, Signal
# 导入字体设置类
from PySide6.QtGui import QFont

# 从同目录的 API 模块导入获取 AI 名称列表的函数
from .API import get_ai_names

# 定义颜色常量，与棋盘保持一致
BLACK = 1
WHITE = 2

class BottomWidget(QWidget):
    """
    底部控制面板类：
    - 显示计时和记分
    - 提供黑棋和白棋的角色选择下拉菜单（玩家或具体AI算法）
    - 提供“开始游戏”和“重新开始”按钮
    - 通过信号与主窗口通信
    """
    # 自定义信号：开始游戏时携带黑棋角色和白棋角色（已映射为 "player" 或算法名）
    start_game_signal = Signal(str, str)
    # 自定义信号：重置游戏（无参数）
    reset_game_signal = Signal()

    def __init__(self, parent=None):
        """构造函数，初始化界面和状态"""
        super().__init__(parent)          # 调用父类 QWidget 的构造函数
        self.init_ui()                    # 构建所有子控件

        # 计分状态
        self.black_wins = 0               # 黑棋累计胜场
        self.white_wins = 0               # 白棋累计胜场

        # 计时状态
        self.elapsed_seconds = 0          # 累计秒数

        # 创建计时器，每秒触发一次更新
        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_timer)   # 超时时调用更新显示

        # 连接按钮的点击事件到内部的槽函数
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)

    def init_ui(self):
        """构建所有界面控件并布局"""
        layout = QVBoxLayout(self)            # 整个底部面板采用垂直布局
        layout.setSpacing(10)                 # 控件间距 10 像素

        # ----- 第一行：计时和记分（水平布局）-----
        info_layout = QHBoxLayout()
        self.timer_label = QLabel("计时：00:00:00")           # 计时标签
        self.timer_label.setFont(QFont("Arial", 16, QFont.Bold))  # type: ignore # 字体加粗
        self.score_label = QLabel("记分：黑棋 0  :  0 白棋")   # 记分标签
        self.score_label.setFont(QFont("Arial", 16, QFont.Bold)) # type: ignore
        info_layout.addWidget(self.timer_label)               # 加入计时
        info_layout.addStretch()                              # 弹簧，将记分推到右边
        info_layout.addWidget(self.score_label)               # 加入记分
        layout.addLayout(info_layout)

        # ----- 第二行：执棋选择（黑棋和白棋下拉菜单）-----
        # 获取所有 AI 算法名称（例如 ["评分算法", "随机算法"]）
        ai_names = get_ai_names()
        # 构建选项列表：第一项是 "玩家"，后面是所有算法名
        choices = ["玩家"] + ai_names

        choice_layout = QHBoxLayout()          # 水平布局容纳两个下拉菜单
        choice_layout.setSpacing(20)
        choice_layout.setAlignment(Qt.AlignCenter)   # type: ignore # 居中对齐

        # 黑棋选择
        self.black_label = QLabel("黑棋：")
        self.black_label.setFont(QFont("Arial", 14))
        self.black_combo = QComboBox()
        self.black_combo.addItems(choices)              # 添加选项
        self.black_combo.setFixedWidth(180)             # 固定宽度
        self.black_combo.setMinimumHeight(40)           # 最小高度，方便触摸点击
        self.black_combo.setFont(QFont("Arial", 14))
        self.black_combo.setCurrentIndex(0)             # 默认选中 "玩家"

        # 白棋选择
        self.white_label = QLabel("白棋：")
        self.white_label.setFont(QFont("Arial", 14))
        self.white_combo = QComboBox()
        self.white_combo.addItems(choices)
        self.white_combo.setFixedWidth(180)
        self.white_combo.setMinimumHeight(40)
        self.white_combo.setFont(QFont("Arial", 14))
        self.white_combo.setCurrentIndex(1)             # 默认选中第一个 AI（索引1）

        choice_layout.addWidget(self.black_label)
        choice_layout.addWidget(self.black_combo)
        choice_layout.addSpacing(30)                    # 两组之间增加间距
        choice_layout.addWidget(self.white_label)
        choice_layout.addWidget(self.white_combo)
        layout.addLayout(choice_layout)

        # ----- 第三行：开始游戏和重新开始按钮 -----
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter) # type: ignore

        self.btn_start = QPushButton("开始游戏/继续")
        self.btn_reset = QPushButton("重新开始(初始化)")
        for btn in (self.btn_start, self.btn_reset):
            # 设置水平拉伸策略，垂直固定
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # type: ignore
            btn.setMinimumWidth(150)          # 最小宽度
            btn.setFixedHeight(50)            # 固定高度
            btn.setFont(QFont("Arial", 14))

        btn_layout.addWidget(self.btn_start, 1)   # 拉伸因子 1，两个按钮等宽
        btn_layout.addWidget(self.btn_reset, 1)
        layout.addLayout(btn_layout)

    # ---------- 内部槽函数（响应按钮点击） ----------
    def on_start_clicked(self):
        """
        点击“开始游戏”按钮时调用。
        将下拉菜单的文本转换成内部角色标识：
        - 如果用户选择的是“玩家”，则映射为字符串 "player"
        - 如果选择的是某个算法名称（如“评分算法”），则原样保留
        然后通过信号发射给主窗口。
        """
        # 获取当前选择的文本
        black_text = self.black_combo.currentText()
        white_text = self.white_combo.currentText()
        # 映射角色：是“玩家”则转为 "player"，否则保留原算法名
        black_role = "player" if black_text == "玩家" else black_text
        white_role = "player" if white_text == "玩家" else white_text
        # 发射开始游戏信号，携带映射后的角色
        self.start_game_signal.emit(black_role, white_role)

    def on_reset_clicked(self):
        """点击“重新开始”按钮时，发射重置信号"""
        self.reset_game_signal.emit()

    # ---------- 外部控制接口（供主窗口调用） ----------
    def start_timer(self):
        """启动计时器（每秒触发一次 update_timer）"""
        self.game_timer.start(1000)   # 1000 毫秒间隔

    def stop_timer(self):
        """停止计时器"""
        self.game_timer.stop()

    def reset_timer(self):
        """重置计时器显示和累计秒数，并停止计时"""
        self.elapsed_seconds = 0
        self.timer_label.setText("计时：00:00:00")
        self.game_timer.stop()        # 停止计时，等待下一次开始

    def update_timer(self):
        """计时器的槽函数：每秒更新一次显示"""
        self.elapsed_seconds += 1
        hours = self.elapsed_seconds // 3600
        minutes = (self.elapsed_seconds % 3600) // 60
        seconds = self.elapsed_seconds % 60
        self.timer_label.setText(f"计时：{hours:02d}:{minutes:02d}:{seconds:02d}")

    def add_win(self, color):
        """
        根据获胜颜色累加胜场，并更新记分显示
        color: BLACK 或 WHITE 常量
        """
        if color == BLACK:
            self.black_wins += 1
        elif color == WHITE:
            self.white_wins += 1
        self.update_score_display()

    def update_score_display(self):
        """更新记分标签的文本"""
        self.score_label.setText(f"记分：黑棋 {self.black_wins}  :  {self.white_wins} 白棋")

    def reset_score(self):
        """重置胜场为零"""
        self.black_wins = 0
        self.white_wins = 0
        self.update_score_display()

    def reset_choices_to_default(self):
        """
        将下拉菜单恢复为默认状态：
        - 黑棋：玩家（索引0）
        - 白棋：第一个AI算法（索引1）
        """
        self.black_combo.setCurrentIndex(0)
        self.white_combo.setCurrentIndex(1)

    def set_controls_enabled(self, enabled):
        """
        启用或禁用下拉菜单和“开始游戏”按钮。
        注意：“重新开始”按钮始终可用，不受此控制。
        enabled: True 启用，False 禁用
        """
        self.black_combo.setEnabled(enabled)
        self.white_combo.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)