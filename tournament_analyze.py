"""
锦标赛分析：读取 tournament_results.json
1. 绘制两两比较热力图（胜率/平均用时/平均落子 × 执黑/执白 = 6 张）
2. 统计每种算法黑棋白棋各自总胜场、平均用时、平均落子
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALGO_NAMES = ["贪心评分", "随机", "Negamax", "MCTS", "DQN"]
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_results.json")
HEATMAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_heatmaps.png")
STATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tournament_stats.json")


def load_results():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_pairwise(results):
    """
    返回 6 个 4×4 矩阵（行=执该色算法，列=对手算法，对角线=NaN）：
    win_b, win_w, time_b, time_w, moves_b, moves_w
    """
    n = len(ALGO_NAMES)
    idx = {name: i for i, name in enumerate(ALGO_NAMES)}

    acc = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            acc[(i, j, "b")] = {"wins": 0.0, "time": 0.0, "moves": 0, "count": 0}
            acc[(i, j, "w")] = {"wins": 0.0, "time": 0.0, "moves": 0, "count": 0}

    for r in results:
        bi, wi = idx[r["black"]], idx[r["white"]]
        winner, t, bm, wm = r["winner"], r["time"], r["black_moves"], r["white_moves"]

        a = acc[(bi, wi, "b")]
        a["count"] += 1; a["time"] += t; a["moves"] += bm
        if winner == 1:
            a["wins"] += 1
        elif winner is None:
            a["wins"] += 0.5

        a = acc[(wi, bi, "w")]
        a["count"] += 1; a["time"] += t; a["moves"] += wm
        if winner == 2:
            a["wins"] += 1
        elif winner is None:
            a["wins"] += 0.5

    mats = {k: np.full((n, n), np.nan) for k in
            ["win_b", "win_w", "time_b", "time_w", "moves_b", "moves_w"]}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for color, wkey, tkey, mkey in [
                ("b", "win_b", "time_b", "moves_b"),
                ("w", "win_w", "time_w", "moves_w"),
            ]:
                a = acc[(i, j, color)]
                if a["count"] > 0:
                    mats[wkey][i, j] = a["wins"] / a["count"]
                    mats[tkey][i, j] = a["time"] / a["count"]
                    mats[mkey][i, j] = a["moves"] / a["count"]
    return mats


def draw_heatmaps(mats):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    n = len(ALGO_NAMES)

    panels = [
        ("胜率（执黑）", mats["win_b"], "{:.1%}", "RdYlGn", 0, 1),
        ("平均用时（执黑）", mats["time_b"], "{:.1f}s", "YlOrRd", None, None),
        ("平均落子（执黑）", mats["moves_b"], "{:.0f}", "YlOrRd", None, None),
        ("胜率（执白）", mats["win_w"], "{:.1%}", "RdYlGn", 0, 1),
        ("平均用时（执白）", mats["time_w"], "{:.1f}s", "YlOrRd", None, None),
        ("平均落子（执白）", mats["moves_w"], "{:.0f}", "YlOrRd", None, None),
    ]

    for ax, (title, data, fmt, cmap, vmin, vmax) in zip(axes.flat, panels):
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax if vmax is not None else np.nanmax(data))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(ALGO_NAMES, rotation=20, ha="right", fontsize=11)
        ax.set_yticklabels(ALGO_NAMES, fontsize=11)
        ax.set_xlabel("对手", fontsize=11)
        ax.set_ylabel("执子算法", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=12, color="gray")
                elif not np.isnan(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                            fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("五子棋算法两两对弈热力图", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(HEATMAP_FILE, dpi=150, bbox_inches="tight")
    print(f"热力图已保存: {HEATMAP_FILE}")
    plt.close()


def compute_final_stats(results):
    idx = {name: i for i, name in enumerate(ALGO_NAMES)}
    stats = {name: {"bw": 0.0, "ww": 0.0, "bg": 0, "wg": 0,
                     "bt": 0.0, "wt": 0.0, "bm": 0, "wm": 0}
             for name in ALGO_NAMES}

    for r in results:
        bn, wn = r["black"], r["white"]
        winner, t = r["winner"], r["time"]

        s = stats[bn]
        s["bg"] += 1; s["bt"] += t; s["bm"] += r["black_moves"]
        if winner == 1: s["bw"] += 1
        elif winner is None: s["bw"] += 0.5

        s = stats[wn]
        s["wg"] += 1; s["wt"] += t; s["wm"] += r["white_moves"]
        if winner == 2: s["ww"] += 1
        elif winner is None: s["ww"] += 0.5

    print("\n" + "=" * 100)
    print("各算法最终统计")
    print("=" * 100)
    hdr = (f"{'算法':<10}│{'黑棋胜场':>8} {'黑棋均时':>9} {'黑棋均子':>8}│"
           f"{'白棋胜场':>8} {'白棋均时':>9} {'白棋均子':>8}")
    print(hdr)
    print("─" * 100)

    out = {}
    for name in ALGO_NAMES:
        s = stats[name]
        bt = s["bt"] / s["bg"] if s["bg"] else 0
        wt = s["wt"] / s["wg"] if s["wg"] else 0
        bm = s["bm"] / s["bg"] if s["bg"] else 0
        wm = s["wm"] / s["wg"] if s["wg"] else 0
        print(f"{name:<10}│{s['bw']:>8.1f} {bt:>8.1f}s {bm:>8.1f}│"
              f"{s['ww']:>8.1f} {wt:>8.1f}s {wm:>8.1f}")
        out[name] = {
            "black_wins": s["bw"], "white_wins": s["ww"],
            "black_games": s["bg"], "white_games": s["wg"],
            "black_avg_time": round(bt, 2), "white_avg_time": round(wt, 2),
            "black_avg_moves": round(bm, 1), "white_avg_moves": round(wm, 1),
        }

    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n统计已保存: {STATS_FILE}")
    return out


def main():
    results = load_results()
    print(f"加载 {len(results)} 局结果")

    mats = compute_pairwise(results)
    draw_heatmaps(mats)
    compute_final_stats(results)


if __name__ == "__main__":
    main()
