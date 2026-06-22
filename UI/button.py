from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QLabel, QComboBox, QSizePolicy)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont

from .API import get_ai_names

BLACK = 1
WHITE = 2

class BottomWidget(QWidget):
    """
    底部控制面板：
    - 显示计时和记分
    - 提供黑棋和白棋的角色选择下拉菜单（玩家或具体AI算法）
    - 提供"开始游戏"和"重新开始"按钮
    """
    start_game_signal = Signal(str, str)
    reset_game_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

        # 计分状态
        self.black_wins = 0
        self.white_wins = 0

        # 计时状态
        self.elapsed_seconds = 0

        self.game_timer = QTimer(self)
        self.game_timer.timeout.connect(self.update_timer)

        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_reset.clicked.connect(self.on_reset_clicked)

    def init_ui(self):
        """构建所有界面控件并布局"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # 第一行：计时和记分
        info_layout = QHBoxLayout()
        self.timer_label = QLabel("计时：00:00:00")
        self.timer_label.setFont(QFont("Arial", 16, QFont.Bold))  # type: ignore
        self.score_label = QLabel("记分：黑棋 0  :  0 白棋")
        self.score_label.setFont(QFont("Arial", 16, QFont.Bold)) # type: ignore
        info_layout.addWidget(self.timer_label)
        info_layout.addStretch()
        info_layout.addWidget(self.score_label)
        layout.addLayout(info_layout)

        # 第二行：执棋选择
        ai_names = get_ai_names()
        choices = ["玩家"] + ai_names

        choice_layout = QHBoxLayout()
        choice_layout.setSpacing(20)
        choice_layout.setAlignment(Qt.AlignCenter)   # type: ignore

        self.black_label = QLabel("黑棋：")
        self.black_label.setFont(QFont("Arial", 14))
        self.black_combo = QComboBox()
        self.black_combo.addItems(choices)
        self.black_combo.setFixedWidth(180)
        self.black_combo.setMinimumHeight(40)
        self.black_combo.setFont(QFont("Arial", 14))
        self.black_combo.setCurrentIndex(0)

        self.white_label = QLabel("白棋：")
        self.white_label.setFont(QFont("Arial", 14))
        self.white_combo = QComboBox()
        self.white_combo.addItems(choices)
        self.white_combo.setFixedWidth(180)
        self.white_combo.setMinimumHeight(40)
        self.white_combo.setFont(QFont("Arial", 14))
        self.white_combo.setCurrentIndex(1)

        choice_layout.addWidget(self.black_label)
        choice_layout.addWidget(self.black_combo)
        choice_layout.addSpacing(30)
        choice_layout.addWidget(self.white_label)
        choice_layout.addWidget(self.white_combo)
        layout.addLayout(choice_layout)

        # 第三行：开始游戏和重新开始按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter) # type: ignore

        self.btn_start = QPushButton("开始游戏/继续")
        self.btn_reset = QPushButton("重新开始(初始化)")
        for btn in (self.btn_start, self.btn_reset):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # type: ignore
            btn.setMinimumWidth(150)
            btn.setFixedHeight(50)
            btn.setFont(QFont("Arial", 14))

        btn_layout.addWidget(self.btn_start, 1)
        btn_layout.addWidget(self.btn_reset, 1)
        layout.addLayout(btn_layout)

    def on_start_clicked(self):
        """点击"开始游戏"按钮：将"玩家"映射为 "player"，算法名原样保留"""
        black_text = self.black_combo.currentText()
        white_text = self.white_combo.currentText()
        black_role = "player" if black_text == "玩家" else black_text
        white_role = "player" if white_text == "玩家" else white_text
        self.start_game_signal.emit(black_role, white_role)

    def on_reset_clicked(self):
        """点击"重新开始"按钮：发射重置信号"""
        self.reset_game_signal.emit()

    def start_timer(self):
        """启动计时器（每秒触发一次 update_timer）"""
        self.game_timer.start(1000)

    def stop_timer(self):
        """停止计时器"""
        self.game_timer.stop()

    def reset_timer(self):
        """重置计时器显示和累计秒数，并停止计时"""
        self.elapsed_seconds = 0
        self.timer_label.setText("计时：00:00:00")
        self.game_timer.stop()

    def update_timer(self):
        """计时器的槽函数：每秒更新一次显示"""
        self.elapsed_seconds += 1
        hours = self.elapsed_seconds // 3600
        minutes = (self.elapsed_seconds % 3600) // 60
        seconds = self.elapsed_seconds % 60
        self.timer_label.setText(f"计时：{hours:02d}:{minutes:02d}:{seconds:02d}")

    def add_win(self, color):
        """根据获胜颜色累加胜场，并更新记分显示"""
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
        """将下拉菜单恢复为默认状态：黑棋=玩家，白棋=第一个AI算法"""
        self.black_combo.setCurrentIndex(0)
        self.white_combo.setCurrentIndex(1)

    def set_controls_enabled(self, enabled):
        """
        启用或禁用下拉菜单和"开始游戏"按钮。
        注意："重新开始"按钮始终可用，不受此控制。
        """
        self.black_combo.setEnabled(enabled)
        self.white_combo.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
