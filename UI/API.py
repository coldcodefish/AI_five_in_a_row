from AI import ai_random, ai_rating, ai_negamax
from AI import ai_mcts
from AI import ai_dqn

AI_ALGORITHMS = {
    "随机算法(AI)": ai_random.get_best_move,
    "贪心评分(AI)": ai_rating.get_best_move,
    "Negamax搜索(AI)": ai_negamax.get_best_move,
    "MCTS(AI)": ai_mcts.get_best_move,
    "强化学习(AI)": ai_dqn.get_best_move,
}

def get_ai_move(board, ai_color, player_color, algorithm):
    """根据算法名称调用对应的AI函数，返回 (row, col) 或 None"""
    if algorithm in AI_ALGORITHMS:
        return AI_ALGORITHMS[algorithm](board, ai_color, player_color)
    else:
        return list(AI_ALGORITHMS.values())[0](board, ai_color, player_color)

def get_ai_names():
    """返回所有算法名称列表"""
    return list(AI_ALGORITHMS.keys())
