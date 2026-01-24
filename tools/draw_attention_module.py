# tools/draw_attention_module.py
# -*- coding: utf-8 -*-

from graphviz import Digraph
import os


def draw_attention():
    # 依然使用竖向紧凑布局
    dot = Digraph(comment='Additive Attention Structure', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.5')

    # 全局样式
    dot.attr('node', shape='box', style='filled,rounded', fillcolor='#E3F2FD',
             fontname='Microsoft YaHei', fontsize='11', height='0.5')
    dot.attr('edge', fontname='Microsoft YaHei', fontsize='10')

    # ================= 节点定义 =================

    # 1. 输入：Bi-GRU 的隐状态序列
    # 为了展示序列感，我们画三个代表性节点
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('H1', '隐状态 H1', fillcolor='#C8E6C9')
        s.node('Hi', '隐状态 Hi', fillcolor='#C8E6C9')
        s.node('Hn', '隐状态 Ht', fillcolor='#C8E6C9')

    # 2. 变换层 (Tanh + Weight)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Tanh1', 'Tanh(Wx+b)', shape='ellipse', fillcolor='#FFF9C4')
        s.node('Tanhi', 'Tanh(Wx+b)', shape='ellipse', fillcolor='#FFF9C4')
        s.node('Tanhn', 'Tanh(Wx+b)', shape='ellipse', fillcolor='#FFF9C4')

    # 3. 得分 (Score)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Score1', '得分 e1', shape='circle', width='0.6', fillcolor='#FFCC80')
        s.node('Scorei', '得分 ei', shape='circle', width='0.6', fillcolor='#FFCC80')
        s.node('Scoren', '得分 et', shape='circle', width='0.6', fillcolor='#FFCC80')

    # 4. Softmax 层 (归一化)
    dot.node('Softmax', 'Softmax\n归一化', fillcolor='#FFAB91', width='2.5')

    # 5. 权重 (Alpha)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Alpha1', '权重 α1', shape='diamond', fillcolor='#F48FB1')
        s.node('Alphai', '权重 αi', shape='diamond', fillcolor='#F48FB1')
        s.node('Alphan', '权重 αt', shape='diamond', fillcolor='#F48FB1')

    # 6. 加权求和 (Sum)
    dot.node('Sum', '加权求和 (Weighted Sum)\nΣ αi · Hi', fillcolor='#CE93D8', width='2.5')

    # 7. 输出
    dot.node('Context', '上下文向量 c\n(Context Vector)', shape='ellipse', fillcolor='#FFF3E0')

    # ================= 连线定义 =================

    # H -> Tanh
    dot.edge('H1', 'Tanh1')
    dot.edge('Hi', 'Tanhi')
    dot.edge('Hn', 'Tanhn')

    # Tanh -> Score
    dot.edge('Tanh1', 'Score1')
    dot.edge('Tanhi', 'Scorei')
    dot.edge('Tanhn', 'Scoren')

    # Score -> Softmax
    dot.edge('Score1', 'Softmax')
    dot.edge('Scorei', 'Softmax')
    dot.edge('Scoren', 'Softmax')

    # Softmax -> Alpha
    dot.edge('Softmax', 'Alpha1')
    dot.edge('Softmax', 'Alphai')
    dot.edge('Softmax', 'Alphan')

    # Alpha + H -> Sum (关键路径：权重和原始状态结合)
    # 使用虚线表示 H 直接连到 Sum，实线表示 Alpha 连到 Sum
    dot.edge('Alpha1', 'Sum')
    dot.edge('Alphai', 'Sum')
    dot.edge('Alphan', 'Sum')

    # 旁路连接：H -> Sum
    dot.edge('H1', 'Sum', style='dashed', color='grey')
    dot.edge('Hi', 'Sum', style='dashed', color='grey')
    dot.edge('Hn', 'Sum', style='dashed', color='grey')

    # Sum -> Output
    dot.edge('Sum', 'Context')

    # ================= 保存 =================
    output_dir = 'results/paper_figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'fig_4-4_attention_module')
    dot.render(output_path, view=False)
    print(f"✅ 注意力机制结构图已生成: {output_path}.png")


if __name__ == "__main__":
    draw_attention()