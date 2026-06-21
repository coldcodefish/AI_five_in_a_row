# UI/board.py
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from .API import get_ai_move      # 导入AI调用接口

BOARD_SIZE = 15                  # 棋盘大小
EMPTY = 0
BLACK = 1
WHITE = 2

class ChessBoard(QWidget):
    """
    棋盘绘制与交互组件：维护棋盘状态，处理鼠标点击，调用AI，判定胜负
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 600)          # 保证棋盘显示完整
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]  # 初始化空棋盘
        self.current_player = BLACK            # 黑棋先手
        self.game_over = False                 # 游戏是否结束
        self.game_started = False              # 是否开始
        # role字典存储每个颜色的角色："player" 或算法名称（如"评分算法"）
        self.role = {BLACK: "player", WHITE: "评分算法"}   # 默认黑玩家，白AI
        self.move_history = []                 # 落子历史（未使用）
        self.winner = None                     # 胜者

        self.ai_timer = QTimer(self)           # AI走棋定时器
        self.ai_timer.setSingleShot(True)      # 只触发一次
        self.ai_timer.timeout.connect(self.ai_move)  # 超时执行ai_move

        self.game_finished = None              # 外部回调函数

    def set_game_finished_callback(self, callback):
        """设置游戏结束时的回调函数，回调参数为winner（颜色或None）"""
        self.game_finished = callback

    def start_game(self, black_role, white_role):
        """
        开始新游戏：传入黑棋和白棋的角色
        black_role / white_role: "player" 或算法名称
        """
        self.role[BLACK] = black_role
        self.role[WHITE] = white_role
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]  # 清空棋盘
        self.move_history = []
        self.current_player = BLACK            # 重置为先手
        self.game_over = False
        self.game_started = True
        self.winner = None
        self.update()                          # 重绘

        # 如果黑棋是AI（角色不是"player"），则触发AI走棋
        if self.role[BLACK] != "player":
            self.ai_timer.start(100)

    def reset_board(self):
        """重置棋盘到未开始状态"""
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.move_history = []
        self.current_player = BLACK
        self.game_over = False
        self.game_started = False
        self.winner = None
        self.ai_timer.stop()
        self.update()

    def paintEvent(self, event):
        """绘制棋盘、网格、星位和所有棋子，以及游戏结束信息"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # type: ignore # 抗锯齿

        margin = 30
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        cell = min(w, h) // (BOARD_SIZE - 1)        # 格子像素大小
        start_x = (self.width() - cell * (BOARD_SIZE - 1)) // 2
        start_y = (self.height() - cell * (BOARD_SIZE - 1)) // 2

        # 背景（木色）
        painter.fillRect(self.rect(), QColor(220, 180, 140))
        pen = QPen(Qt.black, 1) # type: ignore
        painter.setPen(pen)

        # 画网格线
        for i in range(BOARD_SIZE):
            # 横线
            x1 = start_x
            y1 = start_y + i * cell
            x2 = start_x + (BOARD_SIZE - 1) * cell
            y2 = y1
            painter.drawLine(x1, y1, x2, y2)
            # 竖线
            x1 = start_x + i * cell
            y1 = start_y
            x2 = x1
            y2 = start_y + (BOARD_SIZE - 1) * cell
            painter.drawLine(x1, y1, x2, y2)

        # 画星位（天元、小目等）
        star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for (row, col) in star_points:
            x = start_x + col * cell
            y = start_y + row * cell
            painter.setBrush(QBrush(Qt.black)) # type: ignore
            painter.setPen(QPen(Qt.black, 1)) # type: ignore
            painter.drawEllipse(x - 4, y - 4, 8, 8)

        # 画棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == EMPTY:
                    continue
                x = start_x + c * cell
                y = start_y + r * cell
                radius = cell // 2 - 3
                if self.board[r][c] == BLACK:
                    painter.setBrush(QBrush(Qt.black)) # type: ignore
                    painter.setPen(QPen(Qt.black, 1)) # type: ignore
                else:  # WHITE
                    painter.setBrush(QBrush(Qt.white)) # type: ignore
                    painter.setPen(QPen(Qt.black, 1)) # type: ignore
                painter.drawEllipse(x - radius, y - radius, 2 * radius, 2 * radius)
                # 白棋加高光
                if self.board[r][c] == WHITE:
                    painter.setBrush(QBrush(QColor(240, 240, 240)))
                    painter.setPen(QPen(Qt.NoPen)) # type: ignore
                    painter.drawEllipse(x - radius // 2, y - radius // 2, radius // 2, radius // 2)

        # 如果游戏结束，叠加显示胜负信息
        if self.game_over:
            painter.setPen(QPen(Qt.red, 3)) # type: ignore
            painter.setFont(QFont("Arial", 20, QFont.Bold)) # type: ignore
            if self.winner == BLACK:
                msg = "黑棋胜！"
            elif self.winner == WHITE:
                msg = "白棋胜！"
            else:
                msg = "平局"
            painter.drawText(self.rect(), Qt.AlignCenter, msg) # type: ignore

    def mousePressEvent(self, event):
        """鼠标点击事件：玩家落子"""
        if not self.game_started or self.game_over:
            return
        if self.role[self.current_player] != "player":  # 当前走棋方不是玩家
            return

        # 计算点击位置对应的行列
        margin = 30
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        cell = min(w, h) // (BOARD_SIZE - 1)
        start_x = (self.width() - cell * (BOARD_SIZE - 1)) // 2
        start_y = (self.height() - cell * (BOARD_SIZE - 1)) // 2

        x = event.position().x() - start_x
        y = event.position().y() - start_y
        col = round(x / cell)
        row = round(y / cell)
        # 边界和有效性检查
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return
        if self.board[row][col] != EMPTY:
            return
        self.place_stone(row, col, self.current_player)

    def place_stone(self, row, col, color):
        """
        核心落子函数：放置棋子，检查胜负/平局，切换玩家，触发AI
        """
        if self.game_over or not self.game_started:
            return
        if self.board[row][col] != EMPTY or color != self.current_player:
            return

        self.board[row][col] = color
        self.move_history.append((row, col, color))
        self.update()

        # 检查胜利
        if self.check_win(row, col, color):
            self.game_over = True
            self.winner = color
            self.game_started = False
            if self.game_finished:
                self.game_finished(color)      # 通知主窗口
            self.update()
            return

        # 检查平局（棋盘已满）
        if all(self.board[r][c] != EMPTY for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)):
            self.game_over = True
            self.winner = None
            self.game_started = False
            if self.game_finished:
                self.game_finished(None)
            self.update()
            return

        # 切换当前玩家
        self.current_player = WHITE if color == BLACK else BLACK
        self.update()

        # 如果当前走棋方是AI（不是"player"），启动定时器
        if not self.game_over and self.role[self.current_player] != "player":
            self.ai_timer.start(100)

    def check_win(self, row, col, color):
        """检查以(row,col)为中心，四个方向是否有五子连珠"""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == color:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and self.board[r][c] == color:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def ai_move(self):
        """AI走棋，由定时器触发"""
        if self.game_over or not self.game_started:
            return
        if self.role[self.current_player] == "player":  # 当前走棋方是玩家，不执行
            return

        opponent = WHITE if self.current_player == BLACK else BLACK
        algorithm = self.role[self.current_player]      # 获取算法名称
        move = get_ai_move(self.board, self.current_player, opponent, algorithm)
        if move is None:
            return
        row, col = move
        self.place_stone(row, col, self.current_player)