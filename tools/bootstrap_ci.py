#置信区间是结果，是利用Bootstrap产生的分布信息计算出来的，用于量化不确定性。
# Bootstrap（自助法）是一种强大的重采样技术，由布拉德利·埃弗龙提出。它的核心思想是：当我们没有大量的数据来直接做评估时，
# 可以通过对现有的单一数据集进行“有放回地重复抽样”，来模拟从总体中多次抽样的过程，从而估计统计量的分布。
# 具体做法
#基础：假设你有一个原始数据集 D，大小为 N。
# 生成一个Bootstrap样本：从 D 中有放回地随机抽取 N 个样本。这意味着：
# 有些样本可能会被抽到多次。
# 有些样本可能一次都没被抽到。理论上，每次抽样大约有 （1 - 1/e）≈ 63.2% 的原始样本会被抽到至少一次。
# 重复：重复步骤2很多次（例如1000或10000次），从而生成大量（例如B个）不同的Bootstrap样本集：D_boot1, D_boot2, ..., D_bootB。
# 计算统计量：在每个Bootstrap样本集上计算你关心的统计量（例如，模型的准确率 A_i）。这样你就得到了一个由B个统计量值组成的集合 [A1, A2, ..., AB]。
# 在机器学习中的应用（作为“测试集”）
# 在模型评估中，我们通常将数据分为训练集和测试集。但当一个数据集很小，无法奢侈地留出一大块作为测试集时，Bootstrap就非常有用。
# Bootstrap测试集并不是一个固定的集合，而是一种方法。
# 常见的做法是：在每一个Bootstrap样本集 D_boot_i 上训练模型，然后在原始数据集D上评估该模型的性能（记为 A_i）。
# D_boot_i 作为训练集（因为是有放回抽样，它包含了大约63.2%的原始数据和一些重复样本）。
# 原始数据集D 在这里扮演了“测试集”的角色。因为每次评估都在同一个全集 D 上进行，所以结果可比。那些从未被抽到的样本（约36.8%）构成了一个天然的“袋外”测试集，
# 但更常见的做法是直接使用整个 D 来评估以获得更稳定的估计。
# 通过这个过程，你不仅得到了一个平均性能估计（所有 A_i 的平均值），更重要的是，你得到了一个性能指标的分布，这直接引出了下一个概念——置信区间。
#

# 本脚本对“测试集@best_th_val”的 F1 做 bootstrap 置信区间（例如 1000 次）

import os, json, argparse, numpy as np
from sklearn import metrics

def ci_from_preds(y_true, y_prob, threshold, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    f1s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = (y_prob[idx] >= threshold).astype(int)
        f1s.append(metrics.f1_score(y_t, y_p))
    f1s = np.array(f1s)
    lo, hi = np.percentile(f1s, [2.5, 97.5])
    return float(lo), float(hi)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="路径：tune脚本生成的 summary.json")
    ap.add_argument("--preds_npz", required=False, help="可选：包含y_true_test,y_prob_test的npz（若无，就先改tune脚本把二者保存出来）")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    # 你可以把tune脚本里 y_true_test / y_prob_test 顺便保存成 npz，这里直接读取：
    data = np.load(args.preds_npz)
    y_true_test = data["y_true_test"]
    y_prob_test = data["y_prob_test"]

    lo, hi = ci_from_preds(y_true_test, y_prob_test, args.threshold, n_boot=args.n_boot)
    print(f"95% bootstrap CI for F1 @ th={args.threshold:.3f}: [{lo:.4f}, {hi:.4f}]")
