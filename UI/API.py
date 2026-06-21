# UI/API.py
# 导入所有AI算法模块
from AI import ai_random, ai_rating, ai_negamax
from AI import ai_mcts
# from AI import ai_dqn

# 注册所有AI算法，键名为显示名称，值为对应的函数
AI_ALGORITHMS = {
    "贪心评分(AI)": ai_rating.get_best_move,
    "随机算法(AI)": ai_random.get_best_move,
    "Negamax搜索(AI)": ai_negamax.get_best_move,
    "MCTS(AI)": ai_mcts.get_best_move,
    # "强化学习(AI)": ai_dqn.get_best_move,     
}

def get_ai_move(board, ai_color, player_color, algorithm):
    """
    根据算法名称调用对应的AI函数
    参数：
        board: 棋盘
        ai_color: AI颜色
        player_color: 对手颜色
        algorithm: 算法名称（必须是AI_ALGORITHMS中的键）
    返回：AI走法 (row, col) 或 None
    """
    if algorithm in AI_ALGORITHMS:
        return AI_ALGORITHMS[algorithm](board, ai_color, player_color)
    else:
        # 若算法名不存在，回退到第一个算法（防止崩溃）
        return list(AI_ALGORITHMS.values())[0](board, ai_color, player_color)

def get_ai_names():
    """返回所有算法名称列表，用于UI下拉菜单"""
    return list(AI_ALGORITHMS.keys())