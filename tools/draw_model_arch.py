# # # # tools/draw_model_arch_cn_vertical.py
# # # # -*- coding: utf-8 -*-
# # #
# # # from graphviz import Digraph
# # # import os
# # #
# # #
# # # def draw_architecture():
# # #     dot = Digraph(
# # #         comment='DD-CGA Model Architecture',
# # #         format='png'
# # #     )
# # #
# # #     # ================= 全局图属性 =================
# # #     dot.attr(
# # #         rankdir='TB',
# # #         splines='ortho',
# # #         compound='true',
# # #         nodesep='0.4',   # ★ 控制同层节点间距
# # #         ranksep='0.6'    # ★ 控制层间距，防止过长
# # #     )
# # #
# # #     # ================= 全局节点样式 =================
# # #     dot.attr(
# # #         'node',
# # #         shape='box',
# # #         style='filled',
# # #         fillcolor='#E3F2FD',
# # #         fontname='Microsoft YaHei',
# # #         fontsize='11',
# # #         width='2.8'     # ★ 关键：统一宽度，让高度自适应
# # #     )
# # #
# # #     dot.attr(
# # #         'edge',
# # #         fontname='Microsoft YaHei',
# # #         fontsize='9',
# # #         arrowsize='0.7'
# # #     )
# # #
# # #     # ================= 1. 输入与重塑 =================
# # #     dot.node(
# # #         'Input',
# # #         '输入流量\n64维统计特征',
# # #         shape='ellipse',
# # #         fillcolor='#FFF3E0'
# # #     )
# # #
# # #     dot.node(
# # #         'Reshape1',
# # #         '特征重构\n1D → 2D (8×8)',
# # #         style='dashed,filled',
# # #         fillcolor='#FFFFFF'
# # #     )
# # #
# # #     # ================= 2. 空间特征提取 =================
# # #     with dot.subgraph(name='cluster_Spatial') as c:
# # #         c.attr(
# # #             label='空间特征提取模块\n(Spatial Feature Extraction)',
# # #             style='rounded',
# # #             color='grey',
# # #             fontname='Microsoft YaHei'
# # #         )
# # #
# # #         c.node(
# # #             'CNN',
# # #             '2D CNN\n局部空间模式提取',
# # #             fillcolor='#BBDEFB'
# # #         )
# # #
# # #         c.node(
# # #             'CBAM',
# # #             'CBAM 注意力\n通道 + 空间增强',
# # #             fillcolor='#90CAF9'
# # #         )
# # #
# # #         # ★ 保证同层
# # #         c.attr(rank='same')
# # #
# # #     # ================= 3. 序列化 =================
# # #     dot.node(
# # #         'Reshape2',
# # #         '序列化\n行优先扫描',
# # #         style='dashed,filled',
# # #         fillcolor='#FFFFFF'
# # #     )
# # #
# # #     # ================= 4. 时序建模 =================
# # #     with dot.subgraph(name='cluster_Temporal') as c:
# # #         c.attr(
# # #             label='时序建模模块\n(Temporal Modeling)',
# # #             style='rounded',
# # #             color='grey',
# # #             fontname='Microsoft YaHei'
# # #         )
# # #
# # #         c.node(
# # #             'BiGRU',
# # #             '双向 GRU\n时序依赖建模',
# # #             fillcolor='#C8E6C9'
# # #         )
# # #
# # #     # ================= 5. 全局注意力 =================
# # #     with dot.subgraph(name='cluster_Attn') as c:
# # #         c.attr(
# # #             label='全局时序注意力\n(Global Attention)',
# # #             style='rounded',
# # #             color='grey',
# # #             fontname='Microsoft YaHei'
# # #         )
# # #
# # #         c.node(
# # #             'Attn',
# # #             '加性注意力\n关键时间步聚焦',
# # #             fillcolor='#FFCC80'
# # #         )
# # #
# # #     # ================= 6. 输出 =================
# # #     dot.node(
# # #         'FC',
# # #         '全连接层\n+ Dropout'
# # #     )
# # #
# # #     dot.node(
# # #         'Output',
# # #         '分类输出\nSigmoid',
# # #         shape='ellipse',
# # #         fillcolor='#FFF3E0'
# # #     )
# # #
# # #     # ================= 连线（★ 简化 label） =================
# # #     dot.edge('Input', 'Reshape1')
# # #     dot.edge('Reshape1', 'CNN')
# # #     dot.edge('CNN', 'CBAM')
# # #     dot.edge('CBAM', 'Reshape2')
# # #     dot.edge('Reshape2', 'BiGRU')
# # #     dot.edge('BiGRU', 'Attn')
# # #     dot.edge('Attn', 'FC')
# # #     dot.edge('FC', 'Output')
# # #
# # #     # ================= 保存 =================
# # #     output_dir = 'results/paper_figures'
# # #     os.makedirs(output_dir, exist_ok=True)
# # #
# # #     output_path = os.path.join(
# # #         output_dir,
# # #         'fig_model_architecture_cn_vertical'
# # #     )
# # #
# # #     dot.render(output_path, view=False)
# # #     print(f"✅ 模型结构图已生成: {output_path}.png")
# # #
# # #
# # # if __name__ == "__main__":
# # #     draw_architecture()
# #
# #
# # # tools/draw_model_arch_cn_vertical_landscape.py
# # # -*- coding: utf-8 -*-
# #
# # from graphviz import Digraph
# # import os
# #
# #
# # def draw_architecture():
# #     dot = Digraph(
# #         comment='DD-CGA Model Architecture',
# #         format='png'
# #     )
# #
# #     # ================= 全局图属性 =================
# #     dot.attr(
# #         rankdir='TB',          # ★ 结构仍然是竖向
# #         splines='ortho',
# #         compound='true',
# #         nodesep='0.6',         # ★ 增大横向间距 → 画布自然变宽
# #         ranksep='0.8'          # ★ 控制纵向长度
# #     )
# #
# #     # ================= 全局节点样式 =================
# #     dot.attr(
# #         'node',
# #         shape='box',
# #         style='filled',
# #         fillcolor='#E3F2FD',
# #         fontname='Microsoft YaHei',
# #         fontsize='11',
# #         width='3.2'            # ★ 关键：更宽 → 横向画布
# #     )
# #
# #     dot.attr(
# #         'edge',
# #         fontname='Microsoft YaHei',
# #         fontsize='9',
# #         arrowsize='0.7'
# #     )
# #
# #     # ================= 1. 输入层 =================
# #     dot.node(
# #         'Input',
# #         '输入流量特征\n64 维统计向量',
# #         shape='ellipse',
# #         fillcolor='#FFF3E0'
# #     )
# #
# #     dot.node(
# #         'Reshape1',
# #         '特征重构\n1D → 2D\n(8 × 8)',
# #         style='dashed,filled',
# #         fillcolor='#FFFFFF'
# #     )
# #
# #     # ================= 2. 空间特征提取 =================
# #     with dot.subgraph(name='cluster_Spatial') as c:
# #         c.attr(
# #             label='空间特征提取模块\nSpatial Feature Extraction',
# #             style='rounded',
# #             color='grey',
# #             fontname='Microsoft YaHei'
# #         )
# #
# #         c.node(
# #             'CNN',
# #             '二维卷积网络\n(CNN)\n提取局部空间模式',
# #             fillcolor='#BBDEFB'
# #         )
# #
# #         c.node(
# #             'CBAM',
# #             'CBAM 注意力机制\n通道 + 空间增强',
# #             fillcolor='#90CAF9'
# #         )
# #
# #         # ★ 同一层横向排列，避免纵向拉长
# #         c.attr(rank='same')
# #
# #     # ================= 3. 序列化 =================
# #     dot.node(
# #         'Reshape2',
# #         '序列化处理\n按行展开为序列',
# #         style='dashed,filled',
# #         fillcolor='#FFFFFF'
# #     )
# #
# #     # ================= 4. 时序建模 =================
# #     with dot.subgraph(name='cluster_Temporal') as c:
# #         c.attr(
# #             label='时序建模模块\nTemporal Modeling',
# #             style='rounded',
# #             color='grey',
# #             fontname='Microsoft YaHei'
# #         )
# #
# #         c.node(
# #             'BiGRU',
# #             '双向 GRU 网络\n建模长程时序依赖',
# #             fillcolor='#C8E6C9'
# #         )
# #
# #     # ================= 5. 全局注意力 =================
# #     with dot.subgraph(name='cluster_Attn') as c:
# #         c.attr(
# #             label='全局时序注意力模块\nGlobal Attention',
# #             style='rounded',
# #             color='grey',
# #             fontname='Microsoft YaHei'
# #         )
# #
# #         c.node(
# #             'Attn',
# #             '加性注意力机制\n聚焦关键时间步',
# #             fillcolor='#FFCC80'
# #         )
# #
# #     # ================= 6. 输出层 =================
# #     dot.node(
# #         'FC',
# #         '全连接层\nFully Connected\n+ Dropout'
# #     )
# #
# #     dot.node(
# #         'Output',
# #         '分类输出\nSigmoid\n攻击 / 良性',
# #         shape='ellipse',
# #         fillcolor='#FFF3E0'
# #     )
# #
# #     # ================= 连线（简洁、无 label） =================
# #     dot.edge('Input', 'Reshape1')
# #     dot.edge('Reshape1', 'CNN')
# #     dot.edge('CNN', 'CBAM')
# #     dot.edge('CBAM', 'Reshape2')
# #     dot.edge('Reshape2', 'BiGRU')
# #     dot.edge('BiGRU', 'Attn')
# #     dot.edge('Attn', 'FC')
# #     dot.edge('FC', 'Output')
# #
# #     # ================= 保存 =================
# #     output_dir = 'results/paper_figures/model_arh'
# #     os.makedirs(output_dir, exist_ok=True)
# #
# #     output_path = os.path.join(
# #         output_dir,
# #         'fig_model_architecture_cn_landscape'
# #     )
# #
# #     dot.render(output_path, view=False)
# #     print(f"✅ 横向画布 + 竖向结构模型图已生成: {output_path}.png")
# #
# #
# # if __name__ == "__main__":
# #     draw_architecture()
# # tools/draw_model_arch_cn_horizontal_canvas.py
# # -*- coding: utf-8 -*-
#
# from graphviz import Digraph
# import os
#
#
# def draw_architecture():
#     # ================= 1. 创建有向图 =================
#     dot = Digraph(comment='DD-CGA Model Architecture', format='png')
#
#     # 横向画布（Left → Right）
#     dot.attr(
#         rankdir='LR',
#         splines='ortho',
#         nodesep='0.6',   # 控制左右节点间距
#         ranksep='0.8'    # 控制上下层级间距
#     )
#
#     # ================= 2. 全局样式 =================
#     dot.attr(
#         'node',
#         shape='box',
#         style='rounded,filled',
#         fillcolor='#E3F2FD',
#         fontname='Microsoft YaHei',
#         fontsize='12',
#         height='0.9',
#         width='2.4'
#     )
#     dot.attr('edge', fontname='Microsoft YaHei', fontsize='10')
#
#     # ================= 3. 节点定义 =================
#
#     dot.node(
#         'Input',
#         '输入流量\n一维特征向量',
#         shape='ellipse',
#         fillcolor='#FFF3E0'
#     )
#
#     dot.node(
#         'Reshape1',
#         '维度重塑\nReshape\n1D → 2D',
#         style='dashed,rounded,filled',
#         fillcolor='#FFFFFF'
#     )
#
#     # -------- Spatial Block --------
#     with dot.subgraph(name='cluster_Spatial') as c:
#         c.attr(
#             label='空间特征提取模块\n(Spatial Feature Extraction)',
#             style='rounded',
#             color='grey',
#             fontname='Microsoft YaHei'
#         )
#
#         c.attr(rank='same')
#
#         c.node(
#             'CNN',
#             '2D CNN\n局部空间模式提取',
#             fillcolor='#BBDEFB'
#         )
#         c.node(
#             'CBAM',
#             'CBAM 注意力\n通道 + 空间增强',
#             fillcolor='#90CAF9'
#         )
#
#         # 强制竖向关系
#         c.edge('CNN', 'CBAM')
#
#     dot.node(
#         'Reshape2',
#         '序列化\nSerialization\n二维 → 时序',
#         style='dashed,rounded,filled',
#         fillcolor='#FFFFFF'
#     )
#
#     # -------- Temporal Block --------
#     with dot.subgraph(name='cluster_Temporal') as c:
#         c.attr(
#             label='时序建模模块\n(Temporal Modeling)',
#             style='rounded',
#             color='grey',
#             fontname='Microsoft YaHei'
#         )
#
#         c.node(
#             'BiGRU',
#             '双向 GRU\n建模长程时序依赖',
#             fillcolor='#C8E6C9'
#         )
#
#     # -------- Attention Block --------
#     with dot.subgraph(name='cluster_Attn') as c:
#         c.attr(
#             label='全局时序注意力\n(Global Attention)',
#             style='rounded',
#             color='grey',
#             fontname='Microsoft YaHei'
#         )
#
#         c.node(
#             'Attn',
#             '加性注意力\n聚焦关键时间步',
#             fillcolor='#FFCC80'
#         )
#
#     dot.node(
#         'FC',
#         '全连接层\nFC + Dropout'
#     )
#
#     dot.node(
#         'Output',
#         '分类输出\nSigmoid\n攻击 / 良性',
#         shape='ellipse',
#         fillcolor='#FFF3E0'
#     )
#
#     # ================= 4. 主流程连线（强制竖向逻辑） =================
#     dot.edge('Input', 'Reshape1')
#     dot.edge('Reshape1', 'CNN')
#     dot.edge('CBAM', 'Reshape2')
#     dot.edge('Reshape2', 'BiGRU')
#     dot.edge('BiGRU', 'Attn')
#     dot.edge('Attn', 'FC')
#     dot.edge('FC', 'Output')
#
#     # ================= 5. 输出 =================
#     output_dir = 'results/paper_figures/model_arh'
#     os.makedirs(output_dir, exist_ok=True)
#
#     output_path = os.path.join(
#         output_dir,
#         'fig_model_architecture_horizontal_canvas'
#     )
#     dot.render(output_path, view=False)
#
#     print(f"✅ 横向画布 + 竖向结构模型图已生成：{output_path}.png")
#
#
# if __name__ == "__main__":
#     draw_architecture()
#
# tools/draw_model_compact.py
# -*- coding: utf-8 -*-

from graphviz import Digraph
import os


def draw_compact_architecture():
    # rankdir='TB' 表示整体从上到下，但我们会强制内部横向排列
    dot = Digraph(comment='DD-CGA Compact', format='png')

    # === 关键参数设置 ===
    # nodesep: 同一行节点之间的距离
    # ranksep: 不同行之间的距离
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='0.6')

    # 全局样式
    dot.attr('node', shape='box', style='filled', fillcolor='#E3F2FD',
             fontname='Microsoft YaHei', fontsize='12', height='0.6', width='1.5')
    dot.attr('edge', fontname='Microsoft YaHei', fontsize='10')

    # ================= 定义节点 =================

    # Row 1 Nodes
    dot.node('Input', '输入流量\n(1D Vector)', shape='ellipse', fillcolor='#FFF3E0')
    dot.node('Reshape1', '维度重塑\n(1D -> 2D)', style='dashed,filled', fillcolor='#FFFFFF')
    dot.node('CNN', '2D CNN\n(局部特征)', fillcolor='#BBDEFB')

    # Row 2 Nodes
    dot.node('CBAM', 'CBAM 注意力\n(通道+空间)', fillcolor='#90CAF9')
    dot.node('Reshape2', '序列化\n(行优先扫描)', style='dashed,filled', fillcolor='#FFFFFF')
    dot.node('BiGRU', '双向 GRU\n(时序依赖)', fillcolor='#C8E6C9')

    # Row 3 Nodes
    dot.node('Attn', '全局注意力\n(关键时刻)', fillcolor='#FFCC80')
    dot.node('FC', '全连接层\n(Dropout)', shape='box')
    dot.node('Output', '检测结果\n(Sigmoid)', shape='ellipse', fillcolor='#FFF3E0')

    # ================= 强制分行 (关键步骤) =================

    # 第一行：强制这三个点在同一水平线上
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Input')
        s.node('Reshape1')
        s.node('CNN')

    # 第二行
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('CBAM')
        s.node('Reshape2')
        s.node('BiGRU')

    # 第三行
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('Attn')
        s.node('FC')
        s.node('Output')

    # ================= 定义连线 =================

    # 行内连接 (Horizontal)
    dot.edge('Input', 'Reshape1')
    dot.edge('Reshape1', 'CNN')

    # 换行连接 1 (CNN -> CBAM)
    # 这里的 label 可以写在该连接线上
    dot.edge('CNN', 'CBAM', label=' 特征增强')

    # 行内连接
    dot.edge('CBAM', 'Reshape2')
    dot.edge('Reshape2', 'BiGRU')

    # 换行连接 2 (BiGRU -> Attn)
    dot.edge('BiGRU', 'Attn', label=' 隐状态 H')

    # 行内连接
    dot.edge('Attn', 'FC', label=' 上下文 C')
    dot.edge('FC', 'Output', label=' 概率 P')

    # ================= 保存 =================
    output_dir = 'results/paper_figures/model_arh'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'fig_model_compact')
    dot.render(output_path, view=False)
    print(f"✅ 紧凑版模型图已生成: {output_path}.png")


if __name__ == "__main__":
    draw_compact_architecture()