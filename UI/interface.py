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
        self.setWindowIcon(QIcon("src/main_ico.ico"))

        self.resize(1000, 780)
        self.setMinimumSize(800, 780)

        # 居中显示
        screen = QGuiApplication.primaryScreen()
        center = screen.availableGeometry().center()
        self.move(center.x() - self.width() // 2,
                center.y() - self.height() // 2)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)

        self.board_widget = ChessBoard()
        self.board_widget.set_game_finished_callback(self.on_game_finished)
        main_layout.addWidget(self.board_widget, 1)

        self.bottom = BottomWidget()
        main_layout.addWidget(self.bottom)

        self.bottom.start_game_signal.connect(self.on_start_game)
        self.bottom.reset_game_signal.connect(self.on_reset_game)

        self.bottom.set_controls_enabled(True)

    def on_start_game(self, black_role, white_role):
        """响应开始游戏信号：重置棋盘、计时，传递角色，禁用控件，启动计时"""
        self.board_widget.reset_board()
        self.bottom.reset_timer()
        self.board_widget.start_game(black_role, white_role)
        self.bottom.set_controls_enabled(False)
        self.bottom.start_timer()

    def on_reset_game(self):
        """响应重新开始信号：重置棋盘、计时、计分、恢复默认选择、启用控件"""
        self.board_widget.reset_board()
        self.bottom.reset_timer()
        self.bottom.reset_score()
        self.bottom.reset_choices_to_default()
        self.bottom.set_controls_enabled(True)

    def on_game_finished(self, winner):
        """游戏结束回调：停止计时、计时归零、累加胜场、启用控件"""
        self.bottom.stop_timer()
        self.bottom.reset_timer()
        if winner is not None:
            self.bottom.add_win(winner)
        self.bottom.set_controls_enabled(True)
