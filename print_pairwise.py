import numpy as np
from tournament_analyze import load_results, compute_pairwise, ALGO_NAMES

r = load_results()
m = compute_pairwise(r)

def print_mat(key, title):
    n = len(ALGO_NAMES)
    print(f"\n{title}")
    print(f"{'':>10}", end="")
    for j in range(n):
        print(f"{ALGO_NAMES[j]:>12}", end="")
    print()
    for i in range(n):
        print(f"{ALGO_NAMES[i]:<10}", end="")
        for j in range(n):
            v = m[key][i, j]
            if np.isnan(v):
                print(f"{'--':>12}", end="")
            else:
                print(f"{v:>12.1%}" if "win" in key else f"{v:>12.1f}", end="")
        print()

print_mat("win_b", "胜率(执黑) 行=黑方算法 列=白方对手")
print_mat("win_w", "胜率(执白) 行=白方算法 列=黑方对手")
print_mat("time_b", "平均用时秒(执黑) 行=黑方算法 列=白方对手")
print_mat("time_w", "平均用时秒(执白) 行=白方算法 列=黑方对手")
print_mat("moves_b", "平均落子(执黑) 行=黑方算法 列=白方对手")
print_mat("moves_w", "平均落子(执白) 行=白方算法 列=黑方对手")
