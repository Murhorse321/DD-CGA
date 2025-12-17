import pandas as pd
#在划分完成训练验证测试集之后，对其标签分布进行确认
# 数据路径
train_path = "../datas/splits/train_dataset.csv"
val_path = "../datas/splits/val_dataset.csv"
test_path = "../datas/splits/test_dataset.csv"

# 读取数据
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

def check_distribution(df, name):
    counts = df["label"].value_counts()
    total = len(df)
    print(f"\n📊 {name} 数据集:")
    for label, count in counts.items():
        ratio = count / total * 100
        print(f"  标签 {label}: {count} ({ratio:.2f}%)")
    print(f"  总数: {total}")

# 检查各个数据集
check_distribution(train_df, "训练集")
check_distribution(val_df, "验证集")
check_distribution(test_df, "测试集")
