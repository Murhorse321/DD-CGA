import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ======================== 配置区域 ========================

# 根目录（改成你自己的）
BASE_DIR = r'E:\CIC-DDoS\CSVS_chart'

# 要统计的子目录
TARGET_FOLDERS = ['CSV-01-12', 'CSV-03-11']

# 输出图像名称
MULTI_SAVE_PATH = 'cicddos2019_multiclass_distribution.png'
BINARY_SAVE_PATH = 'cicddos2019_binary_distribution.png'

# 绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 标签规范化 ========================

LABEL_NORMALIZATION_MAP = {
    'UDP-lag': 'UDP-lag',
    'UDPLag': 'UDP-lag',

    'UDP': 'DrDoS_UDP',
    'MSSQL': 'DrDoS_MSSQL',
    'LDAP': 'DrDoS_LDAP',

    'SYN': 'Syn',
    'Syn': 'Syn',

    'BENIGN': 'BENIGN',
    'Benign': 'BENIGN'
}


def to_binary_label(label):
    return 'Benign' if label == 'BENIGN' else 'Attack'


# ======================== 工具函数 ========================

def find_label_column(csv_path):
    """自动查找标签列"""
    try:
        df_head = pd.read_csv(csv_path, nrows=0)
        for col in df_head.columns:
            if 'label' in col.lower():
                return col
        return None
    except Exception:
        return None


def collect_all_csv_files():
    """递归收集所有 CSV 文件"""
    all_files = []
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 文件夹不存在: {folder_path}")
            continue

        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith('.csv'):
                    all_files.append(os.path.join(root, f))

    print(f"🔎 共发现 {len(all_files)} 个 CSV 文件")
    return all_files


def count_classes(binary=False):
    """统计类别分布"""
    total_dist = pd.Series(dtype=int)
    all_files = collect_all_csv_files()

    for file_path in tqdm(all_files, desc="Processing CSVs", unit="file"):
        label_col = find_label_column(file_path)
        if not label_col:
            continue

        try:
            df = pd.read_csv(file_path, usecols=[label_col])
            df[label_col] = df[label_col].astype(str).str.strip()
            df[label_col] = df[label_col].replace(LABEL_NORMALIZATION_MAP)

            if binary:
                df['BinaryLabel'] = df[label_col].apply(to_binary_label)
                counts = df['BinaryLabel'].value_counts()
            else:
                counts = df[label_col].value_counts()

            total_dist = total_dist.add(counts, fill_value=0)

        except Exception as e:
            print(f"❌ 读取失败: {os.path.basename(file_path)} -> {e}")

    return total_dist


def plot_distribution(dist, title, save_path):
    """绘制分布图"""
    df = dist.reset_index()
    df.columns = ['Class', 'Count']
    df = df.sort_values('Count', ascending=False)

    print(f"\n📊 {title}")
    print(df)
    print(f"总样本数: {int(df['Count'].sum()):,}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Count', y='Class', data=df, ax=ax)

    ax.set_xscale('log')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Number of Samples (Log Scale)')
    ax.set_ylabel('Traffic Class')

    for i, v in enumerate(df['Count']):
        ax.text(v * 1.05, i, f"{int(v):,}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 图已保存: {os.path.abspath(save_path)}")


# ======================== 主程序 ========================

if __name__ == "__main__":

    # -------- 多分类（仅用于分析） --------
    multi_dist = count_classes(binary=False)
    plot_distribution(
        multi_dist,
        'CIC-DDoS2019 Traffic Distribution (Multi-class, Normalized)',
        MULTI_SAVE_PATH
    )

    # -------- 二分类（最终训练） --------
    binary_dist = count_classes(binary=True)
    plot_distribution(
        binary_dist,
        'CIC-DDoS2019 Traffic Distribution (Binary Classification)',
        BINARY_SAVE_PATH
    )
