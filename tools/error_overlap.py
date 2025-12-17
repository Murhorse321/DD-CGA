# 证明 GRU 的错与 CNN 并非完全重合，存在“关键信息被 last 聚合稀释”的可能 → 自然引出 Attention。
# 在同一测试集上比较两模型错误集合的 Jaccard 重叠度为 J，说明错误模式存在一定互补性。
# 考虑到 GRU 使用 last 聚合，关键信号可能被时序聚合稀释，因而在下一步引入注意力聚合以缓解该问题。
import argparse, pandas as pd, numpy as np
def load(p):
    df = pd.read_csv(p)
    cols = {c.strip().lower(): c for c in df.columns}
    return df[cols["y_true"]].to_numpy().astype(int), df[cols["y_pred"]].to_numpy().astype(int)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)  # CNN test_preds.csv
    ap.add_argument("--b", required=True)  # GRU test_preds.csv
    args = ap.parse_args()
    ya, pa = load(args.a); yb, pb = load(args.b)
    assert (ya == yb).all(), "需要同一测试集（配对）"
    y = ya
    ca, cb = (pa == y), (pb == y)
    n_cc = int((ca & cb).sum()); n_ce = int((ca & ~cb).sum())
    n_ec = int((~ca & cb).sum()); n_ee = int((~ca & ~cb).sum())
    Ea = set(np.where(~ca)[0]); Eb = set(np.where(~cb)[0])
    inter, union = len(Ea & Eb), len(Ea | Eb) or 1
    print("both_correct", n_cc, "| A_correct_B_error", n_ce,
          "| A_error_B_correct", n_ec, "| both_error", n_ee)
    print(f"error Jaccard={inter/union:.4f}  (0=互补, 1=重叠)")
# both_correct 295308 | A_correct_B_error 1932 | A_error_B_correct 1205 | both_error 1555
# error Jaccard=0.3314  (0=互补, 1=重叠)