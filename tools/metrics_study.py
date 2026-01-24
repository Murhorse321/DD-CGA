import sys


class MetricLearner:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def calculate_all(self):
        """基于当前的混淆矩阵计算所有指标"""
        total = self.tp + self.tn + self.fp + self.fn

        # 防止除以0的错误
        if total == 0:
            return "数据为空，无法计算"

        # 1. Accuracy (准确率)
        acc = (self.tp + self.tn) / total

        # 2. Precision (精确率)
        # 只有当 (TP + FP) > 0 时才有意义
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

        # 3. Recall (召回率)
        # 只有当 (TP + FN) > 0 时才有意义
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

        # 4. F1 Score
        # 只有当 (Precision + Recall) > 0 时才有意义
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }

    def print_metrics(self):
        metrics = self.calculate_all()
        print("-" * 30)
        print(f"当前数据: TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}")
        print("-" * 30)
        if isinstance(metrics, str):
            print(metrics)
        else:
            print(f"【Accuracy  准确率】: {metrics['Accuracy']:.4f} (整体猜对的比例)")
            print(f"【Precision 精确率】: {metrics['Precision']:.4f} (猜是正类，实际也是正类的比例)")
            print(f"【Recall    召回率】: {metrics['Recall']:.4f}    (正类样本被抓出来的比例)")
            print(f"【F1 Score  综合分】: {metrics['F1_Score']:.4f}  (P和R的平衡点)")
        print("-" * 30)

    def solve_missing_triangle(self, known_type, val1, val2):
        """
        处理 Precision, Recall, F1 之间的三角关系。
        注意：Accuracy无法通过这三个直接推导（因为它包含TN，而这三个不包含TN）。
        """
        print(f"\n>>> 正在尝试根据 {known_type} 推导其余值...")

        try:
            if known_type == "P_R":  # 已知 P 和 R，求 F1
                p, r = val1, val2
                f1 = 2 * p * r / (p + r)
                print(f"已知 Precision={p}, Recall={r} -> 计算得出 F1={f1:.4f}")

            elif known_type == "F1_P":  # 已知 F1 和 P，求 Recall
                f1, p = val1, val2
                # 公式推导: F1 = 2PR / (P+R)  =>  F1(P+R) = 2PR => F1*P + F1*R = 2PR => F1*P = R(2P - F1)
                if (2 * p - f1) == 0:
                    print("数值导致分母为0，无法计算")
                else:
                    r = (f1 * p) / (2 * p - f1)
                    print(f"已知 F1={f1}, Precision={p} -> 计算得出 Recall={r:.4f}")

            elif known_type == "F1_R":  # 已知 F1 和 R，求 Precision
                f1, r = val1, val2
                if (2 * r - f1) == 0:
                    print("数值导致分母为0，无法计算")
                else:
                    p = (f1 * r) / (2 * r - f1)
                    print(f"已知 F1={f1}, Recall={r} -> 计算得出 Precision={p:.4f}")

            print("注意：Accuracy 无法仅通过 P/R/F1 计算出来，必须知道 TN (真负类) 的数量。")

        except Exception as e:
            print(f"计算出错: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    print("欢迎使用指标学习助手！")

    # 模式 1: 基础计算
    print("\n【模式 1: 基于混淆矩阵计算】")
    try:
        tp = int(input("请输入 TP (真阳性): "))
        tn = int(input("请输入 TN (真阴性): "))
        fp = int(input("请输入 FP (假阳性/误报): "))
        fn = int(input("请输入 FN (假阴性/漏报): "))

        learner = MetricLearner(tp, tn, fp, fn)
        learner.print_metrics()
    except ValueError:
        print("请输入整数！")

    # 模式 2: 关系推导
    print("\n【模式 2: 指标反推 (针对 Precision, Recall, F1)】")
    print("如果你修改其中一项，我们可以计算其他的（但仅限于这三者之间）。")
    print("1. 已知 Precision 和 Recall -> 求 F1")
    print("2. 已知 F1 和 Precision -> 求 Recall")
    print("3. 已知 F1 和 Recall -> 求 Precision")

    choice = input("请选择 (1/2/3): ")

    try:
        if choice == '1':
            p = float(input("输入 Precision (0-1): "))
            r = float(input("输入 Recall (0-1): "))
            learner.solve_missing_triangle("P_R", p, r)
        elif choice == '2':
            f1 = float(input("输入 F1 (0-1): "))
            p = float(input("输入 Precision (0-1): "))
            learner.solve_missing_triangle("F1_P", f1, p)
        elif choice == '3':
            f1 = float(input("输入 F1 (0-1): "))
            r = float(input("输入 Recall (0-1): "))
            learner.solve_missing_triangle("F1_R", f1, r)
    except ValueError:
        print("请输入有效的小数！")