# tools/draw_cbam_square.py
# -*- coding: utf-8 -*-

from graphviz import Digraph
import os


def draw_cbam_square():
    # rankdir='TB' 依然维持整体从上到下的流向
    # splines='ortho' 使用折线，看起来更整洁
    dot = Digraph(comment='CBAM Square Structure', format='png')

    # 【关键调整】缩小节点和层级间距，使图更紧凑
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.4', ranksep='0.5')

    # 全局样式：使用较小的字体和高度，节省空间
    dot.attr('node', shape='box', style='filled,rounded', fillcolor='#E3F2FD',
             fontname='Microsoft YaHei', fontsize='11', height='0.4', width='1.2')
    dot.attr('edge', fontname='Microsoft YaHei', fontsize='10', arrowhead='vee')

    # ================= 1. 定义主干节点 =================
    dot.node('Input', '输入特征图 F', shape='box', style='filled', fillcolor='#FFF3E0', width='1.5')
    # 中间节点用虚线表示临时状态
    dot.node('Refined1', '通道增强特征 F\'', shape='box', style='dashed,filled', fillcolor='#FFFFFF', width='1.5')
    dot.node('Output', '最终输出特征 F\'\'', shape='box', style='filled', fillcolor='#FFF3E0', width='1.5')

    # ================= 2. 通道注意力模块 (CAM) - 蓝色系 =================
    with dot.subgraph(name='cluster_CAM') as c:
        c.attr(style='rounded,dashed', color='#1565C0', label='(a) 通道注意力模块 (Channel Attention)',
               fontname='Microsoft YaHei', fontcolor='#1565C0', bgcolor='#E8EAF6')

        # 定义内部节点
        c.node('MaxPool1', '全局最大池化', fillcolor='#BBDEFB')
        c.node('AvgPool1', '全局平均池化', fillcolor='#BBDEFB')
        # 将 MLP 和 Sum 放在一起以节省空间
        c.node('MLP_Sum', '共享 MLP \n+ 逐元素相加', fillcolor='#90CAF9', height='0.5')
        c.node('Sigmoid1', 'Sigmoid\n激活', fillcolor='#64B5F6', width='0.8')
        # 乘法节点用圆形
        c.node('Mul1', '×', shape='circle', fillcolor='#FFCC80', width='0.4', height='0.4', fontsize='14',
               style='filled')

        # 【关键布局技巧 1】强制并行，增加宽度
        # 将两个池化操作放在同一行
        with c.subgraph() as s:
            s.attr(rank='same')
            s.node('MaxPool1')
            s.node('AvgPool1')

        # 内部连线
        c.edge('MaxPool1', 'MLP_Sum')
        c.edge('AvgPool1', 'MLP_Sum')
        c.edge('MLP_Sum', 'Sigmoid1')
        c.edge('Sigmoid1', 'Mul1', label=' 通道权重 Mc')

    # ================= 3. 空间注意力模块 (SAM) - 绿色系 =================
    with dot.subgraph(name='cluster_SAM') as c:
        c.attr(style='rounded,dashed', color='#2E7D32', label='(b) 空间注意力模块 (Spatial Attention)',
               fontname='Microsoft YaHei', fontcolor='#2E7D32', bgcolor='#E8F5E9')

        # 定义内部节点
        # 将两个通道池化合并为一个节点描述，节省空间
        c.node('Pool2', '通道轴\n最大+平均池化', fillcolor='#C8E6C9')
        c.node('Concat_Conv', '拼接 (Concat)\n+ 7x7 卷积', fillcolor='#A5D6A7', height='0.5')
        c.node('Sigmoid2', 'Sigmoid\n激活', fillcolor='#66BB6A', width='0.8')
        c.node('Mul2', '×', shape='circle', fillcolor='#FFCC80', width='0.4', height='0.4', fontsize='14',
               style='filled')

        # 【关键布局技巧 2】强制并行
        # 这里为了让图更方，我们将主要操作横向排列
        with c.subgraph() as s:
            s.attr(rank='same')
            s.node('Pool2')
            s.node('Concat_Conv')
            s.node('Sigmoid2')

        # 内部连线 (横向)
        c.edge('Pool2', 'Concat_Conv')
        c.edge('Concat_Conv', 'Sigmoid2')
        # Sigmoid 向下连接到乘法器
        c.edge('Sigmoid2', 'Mul2', label=' 空间权重 Ms')

    # ================= 4. 整体主干连接 =================

    # Input -> CAM 的并行池化层
    dot.edge('Input', 'MaxPool1')
    dot.edge('Input', 'AvgPool1')

    # Input -> CAM 的乘法器 (旁路连接)
    # 使用 constraint=false 让这条线自由连接，不影响主布局
    dot.edge('Input', 'Mul1', label=' 原始特征', style='dashed', constraint='false')

    # CAM 输出 -> Refined1
    dot.edge('Mul1', 'Refined1')

    # Refined1 -> SAM 的池化层
    dot.edge('Refined1', 'Pool2')

    # Refined1 -> SAM 的乘法器 (旁路连接)
    dot.edge('Refined1', 'Mul2', label=' 一次增强特征', style='dashed', constraint='false')

    # SAM 输出 -> 最终输出
    dot.edge('Mul2', 'Output')

    # ================= 5. 保存 =================
    output_dir = 'results/paper_figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'fig_4-2_cbam_square')
    dot.render(output_path, view=False)
    print(f"✅ 紧凑方形 CBAM 结构图已生成: {output_path}.png")


if __name__ == "__main__":
    draw_cbam_square()