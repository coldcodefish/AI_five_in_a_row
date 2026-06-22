"""
五子棋算法两两对弈锦标赛
4 个算法，每对执黑 50 局 + 执白 50 局，共 600 局
记录每局：获胜方、用时、黑白落子数 → tournament_results.json
支持断点续跑（读取已有结果跳过已完成局）
"""
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AI import ai_rating, ai_random, ai_negamax, ai_mcts, ai_dqn

BOARD_SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = 2

GAMES_PER_MATCHUP = 50

ALGORITHMS = {
    "贪心评分": ai_rating.get_best_move,
    "随机":   ai_random.get_best_move,
    "Negamax": ai_negamax.get_best_move,
    "MCTS":   ai_mcts.get_best_move,
    "DQN":    ai_dqn.get_best_move,
}

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_results.json")


def check_win(board, row, col, color):
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            count += 1; r += dr; c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == color:
            count += 1; r -= dr; c -= dc
        if count >= 5:
            return True
    return False


def play_game(black_func, white_func):
    """下一局，返回 (winner, elapsed, black_moves, white_moves)"""
    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    current = BLACK
    bm = wm = 0
    start = time.time()
    while True:
        try:
            if current == BLACK:
                move = black_func(board, BLACK, WHITE)
            else:
                move = white_func(board, WHITE, BLACK)
        except Exception:
            winner = WHITE if current == BLACK else BLACK
            return winner, time.time() - start, bm, wm

        if move is None:
            return None, time.time() - start, bm, wm

        r, c = move
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE) or board[r][c] != EMPTY:
            winner = WHITE if current == BLACK else BLACK
            return winner, time.time() - start, bm, wm

        board[r][c] = current
        if current == BLACK:
            bm += 1
        else:
            wm += 1

        if check_win(board, r, c, current):
            return current, time.time() - start, bm, wm
        if bm + wm >= BOARD_SIZE * BOARD_SIZE:
            return None, time.time() - start, bm, wm

        current = WHITE if current == BLACK else BLACK


def main():
    algo_names = list(ALGORITHMS.keys())

    # 断点续跑
    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    # 预热 JIT
    print("预热 JIT 编译...", flush=True)
    wb = [[0] * 15 for _ in range(15)]
    wb[7][7] = 1; wb[7][8] = 2; wb[8][7] = 1
    for name, func in ALGORITHMS.items():
        try:
            func(wb, 1, 2)
        except Exception:
            pass
    print("预热完成", flush=True)

    # 生成对局槽位：(黑方, 白方)
    slots = []
    for bn in algo_names:
        for wn in algo_names:
            if bn == wn:
                continue
            for _ in range(GAMES_PER_MATCHUP):
                slots.append((bn, wn))

    total = len(slots)
    start_from = len(results)
    if start_from >= total:
        print(f"已完成全部 {total} 局，可直接运行分析")
        return

    print(f"总计 {total} 局，已完成 {start_from}，从第 {start_from + 1} 局开始", flush=True)
    t0 = time.time()

    for idx in range(start_from, total):
        bn, wn = slots[idx]
        winner, elapsed, bm, wm = play_game(ALGORITHMS[bn], ALGORITHMS[wn])

        results.append({
            "black": bn, "white": wn,
            "winner": winner,          # 1=黑胜, 2=白胜, null=平
            "time": round(elapsed, 3),
            "black_moves": bm, "white_moves": wm,
        })

        # 每 5 局存盘
        if (idx + 1) % 5 == 0 or idx == total - 1:
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

        done = idx + 1
        el = time.time() - t0
        rate = done - start_from
        eta = el / rate * (total - done) if rate > 0 else 0
        ws = "黑胜" if winner == 1 else ("白胜" if winner == 2 else "平")
        print(f"[{done}/{total}] {bn}(黑) vs {wn}(白) -> {ws} | {elapsed:.1f}s | {bm}+{wm}手 | ETA:{eta:.0f}s", flush=True)

    print(f"\n全部完成！总耗时 {time.time() - t0:.0f}s")
    print(f"结果已保存到 {RESULTS_FILE}")


if __name__ == "__main__":
    main()
