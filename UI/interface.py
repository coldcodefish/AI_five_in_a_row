# UI/interface.py
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from PySide6.QtGui import QFont, QIcon
from .board import ChessBoard
from .button import BottomWidget
from PySide6.QtGui import QGuiApplication

class MainWindow(QMainWindow):
    """主窗口：组装棋盘和底部控制区，连接信号"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 五子棋")
        # 窗口左上角与任务栏图
        self.setWindowIcon(QIcon("src/main_ico.ico"))
        # # 全局默认字体：英文用 Times New Roman，中文用宋体
        # global_font = QFont(["Times New Roman", "SimSun"])
        # # 应用全局字体
        # self.setFont(global_font)
        
        self.resize(1000, 780)                   # 默认窗口大小
        self.setMinimumSize(800, 780)            # 最小尺寸

        # 居中显示
        screen = QGuiApplication.primaryScreen()
        center = screen.availableGeometry().center()
        self.move(center.x() - self.width() // 2,
                center.y() - self.height() // 2)

        central = QWidget()                      # 中心容器
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)       # 垂直主布局
        main_layout.setSpacing(12)

        # 棋盘组件
        self.board_widget = ChessBoard()
        self.board_widget.set_game_finished_callback(self.on_game_finished)  # 注册回调
        main_layout.addWidget(self.board_widget, 1)   # 拉伸因子1

        # 底部控制组件
        self.bottom = BottomWidget()
        main_layout.addWidget(self.bottom)

        # 连接底部信号到槽函数
        self.bottom.start_game_signal.connect(self.on_start_game)
        self.bottom.reset_game_signal.connect(self.on_reset_game)

        # 初始启用控件
        self.bottom.set_controls_enabled(True)

    # ---------- 信号处理 ----------
    def on_start_game(self, black_role, white_role):
        """响应开始游戏信号：重置棋盘、计时，传递角色，禁用控件，启动计时"""
        self.board_widget.reset_board()            # 清空棋盘
        self.bottom.reset_timer()                  # 计时归零
        self.board_widget.start_game(black_role, white_role)  # 开始对局
        self.bottom.set_controls_enabled(False)    # 禁用下拉和开始按钮
        self.bottom.start_timer()                  # 开始计时

    def on_reset_game(self):
        """响应重新开始信号：重置棋盘、计时、计分、恢复默认选择、启用控件"""
        self.board_widget.reset_board()            # 清空棋盘
        self.bottom.reset_timer()                  # 计时归零
        self.bottom.reset_score()                  # 计分归零
        self.bottom.reset_choices_to_default()     # 恢复默认角色
        self.bottom.set_controls_enabled(True)     # 启用所有控件

    def on_game_finished(self, winner):
        """游戏结束回调：停止计时、计时归零、累加胜场、启用控件"""
        self.bottom.stop_timer()                   # 停止计时
        self.bottom.reset_timer()                  # 计时显示归零
        if winner is not None:                     # 非平局
            self.bottom.add_win(winner)            # 累加胜场
        self.bottom.set_controls_enabled(True)     # 启用控件