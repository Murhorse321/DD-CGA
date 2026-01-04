import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ======================== 配置区域 ========================

# 根目录（请确保此路径正确指向包含 CSV-01-12 和 CSV-03-11 的父目录）
BASE_DIR = r'E:\CIC-DDoS\CSVS_chart'

# 要统计的子目录
TARGET_FOLDERS = ['CSV-01-12', 'CSV-03-11']

# 输出图像名称
MULTI_SAVE_PATH = 'cicddos2019_merged_multiclass_distribution.png'
BINARY_SAVE_PATH = 'cicddos2019_merged_binary_distribution.png'

# 绘图风格设置（符合学术出版要求）
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 适配中文显示
plt.rcParams['axes.unicode_minus'] = False

# ======================== 标签规范化映射表 ========================
# 核心修改：将 DrDoS_X 和 X 进行语义合并
LABEL_NORMALIZATION_MAP = {
    # === 1. MSSQL 合并 ===
    'MSSQL': 'MSSQL',
    'DrDoS_MSSQL': 'MSSQL',

    # === 2. LDAP 合并 ===
    'LDAP': 'LDAP',
    'DrDoS_LDAP': 'LDAP',

    # === 3. NetBIOS 合并 ===
    'NetBIOS': 'NetBIOS',
    'DrDoS_NetBIOS': 'NetBIOS',

    # === 4. UDP 合并 ===
    'UDP': 'UDP',
    'DrDoS_UDP': 'UDP',

    # === 5. UDP-Lag 合并 ===
    'UDP-lag': 'UDP-Lag',
    'UDPLag': 'UDP-Lag',

    # === 6. 其他格式规范化 ===
    'SYN': 'Syn',
    'Syn': 'Syn',
    'BENIGN': 'Benign',
    'Benign': 'Benign',

    # === 7. 保持原样的类别 (显式列出以防遗漏) ===
    'TFTP': 'TFTP',
    'DrDoS_SNMP': 'DrDoS_SNMP',
    'DrDoS_DNS': 'DrDoS_DNS',
    'DrDoS_SSDP': 'DrDoS_SSDP',
    'DrDoS_NTP': 'DrDoS_NTP',
    'Portmap': 'Portmap',
    'WebDDoS': 'WebDDoS'
}


def normalize_label(label):
    """
    对读取到的标签进行清洗和映射
    """
    if not isinstance(label, str):
        return str(label)

    # 去除首尾空格
    label = label.strip()

    # 查表映射，如果在表中则替换，否则保留原名（方便发现未处理的新标签）
    return LABEL_NORMALIZATION_MAP.get(label, label)


def to_binary_label(label):
    """二分类转换"""
    # 只有标准化后的 Benign 视为良性，其余均为攻击
    return 'Benign' if label == 'Benign' else 'Attack'


# ======================== 工具函数 ========================

def find_label_column(csv_path):
    """自动查找标签列（处理列名中可能存在的空格）"""
    try:
        # 只读取表头
        df_head = pd.read_csv(csv_path, nrows=0)
        cols = df_head.columns.tolist()

        # 常见变体优先匹配
        candidates = [' Label', 'Label', 'label', ' label']
        for c in candidates:
            if c in cols:
                return c

        # 模糊匹配
        for col in cols:
            if 'label' in col.lower():
                return col
        return None
    except Exception:
        return None


def collect_all_csv_files():
    """递归收集所有 CSV 文件"""
    all_files = []
    print(f"📂 正在搜索目录: {BASE_DIR}")
    for folder in TARGET_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ 警告: 子文件夹不存在 -> {folder_path}")
            continue

        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith('.csv'):
                    all_files.append(os.path.join(root, f))

    print(f"🔎 共发现 {len(all_files)} 个 CSV 文件待处理")
    return all_files


def count_classes(binary=False):
    """统计类别分布的核心逻辑"""
    total_dist = pd.Series(dtype=int)
    all_files = collect_all_csv_files()

    if not all_files:
        print("❌ 未找到文件，请检查 BASE_DIR 路径是否正确！")
        return pd.Series()

    for file_path in tqdm(all_files, desc="正在统计 CSV 文件", unit="file"):
        label_col = find_label_column(file_path)
        if not label_col:
            print(f"⚠️ 跳过（无标签列）: {os.path.basename(file_path)}")
            continue

        try:
            # 仅读取标签列，极大提升速度
            df = pd.read_csv(file_path, usecols=[label_col])

            # 1. 基础清洗
            raw_labels = df[label_col].astype(str)

            # 2. 映射归一化
            normalized_labels = raw_labels.apply(normalize_label)

            # 3. 统计
            if binary:
                binary_labels = normalized_labels.apply(to_binary_label)
                counts = binary_labels.value_counts()
            else:
                counts = normalized_labels.value_counts()

            total_dist = total_dist.add(counts, fill_value=0)

        except Exception as e:
            print(f"❌ 读取失败: {os.path.basename(file_path)} -> {e}")

    return total_dist


def plot_distribution(dist, title, save_path):
    """绘制分布图"""
    if dist.empty:
        print(f"⚠️ 数据为空，无法绘制: {title}")
        return

    df = dist.reset_index()
    df.columns = ['Class', 'Count']
    df = df.sort_values('Count', ascending=False)

    print(f"\n📊 === 统计结果: {title} ===")
    print(df)
    print(f"总样本数: {int(df['Count'].sum()):,}")
    print("================================")

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Count', y='Class', data=df, ax=ax, palette='viridis', edgecolor='black')

    # 设置对数坐标，因为类别不平衡极严重
    ax.set_xscale('log')

    ax.set_title(title, fontsize=15, pad=15)
    ax.set_xlabel('Sample Count (Log Scale)', fontsize=12)
    ax.set_ylabel('Traffic Category', fontsize=12)

    # 在柱子旁显示具体数值
    for i, v in enumerate(df['Count']):
        ax.text(v * 1.1, i, f"{int(v):,}", va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✅ 图片已保存至: {os.path.abspath(save_path)}")


# ======================== 主程序 ========================

if __name__ == "__main__":
    print("🚀 开始执行 CIC-DDoS2019 数据集类别合并与统计...\n")

    # -------- 1. 多分类统计（合并后） --------
    print("--- 正在进行多分类统计 ---")
    multi_dist = count_classes(binary=False)
    plot_distribution(
        multi_dist,
        'CIC-DDoS2019 Distribution (Merged Categories)',
        MULTI_SAVE_PATH
    )

    # -------- 2. 二分类统计 --------
    print("\n--- 正在进行二分类统计 ---")
    binary_dist = count_classes(binary=True)
    plot_distribution(
        binary_dist,
        'CIC-DDoS2019 Binary Distribution (Benign vs Attack)',
        BINARY_SAVE_PATH
    )