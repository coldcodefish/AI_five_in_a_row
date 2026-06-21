import sys                      # 导入系统模块，用于命令行参数
from PySide6.QtWidgets import QApplication   # 导入Qt应用程序类
from UI.interface import MainWindow          # 从UI包导入主窗口类

if __name__ == "__main__":      # 判断是否作为主程序运行
    app = QApplication(sys.argv)  # 创建Qt应用程序实例，传入命令行参数
    window = MainWindow()        # 创建主窗口实例
    window.show()               # 显示主窗口
    sys.exit(app.exec())        # 进入事件循环，程序退出时安全结束