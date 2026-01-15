import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['hatch.linewidth'] = 1.2
mpl.rcParams['hatch.color'] = 'black'

# 更新标签和数据
labels = ['SR-ReAct', 'SR-Transferred', 'GC-ReAct', 'GC-Transferred']


alfred_values = [44.4, 48.8, 50.4, 57.0]

habitat_values = [43.6, 46.4, 50.4, 54.2]

'''
alfred_values = [41.2, 46.4, 47.5, 51.9]
habitat_values = [41.6, 54.2, 47.4, 51.9]
'''
fig, ax = plt.subplots(figsize=(8, 6), dpi=100) # 画布稍微调宽一点，因为加了间隙

colors_refusal = ['#f1c8c2', '#fee5d4', '#f3d0a9','#bfd3c2']
hatch_patterns = ['/', '+', '\\', '.']
edge_color = 'black'
edge_width = 2.5
alpha_refusal = 1.0
separator_color = '#CCCCCC'

# 字体设置
GLOBAL_BASE_FONT_SIZE = 14
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 8
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 1
ANNOTATION_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2

# === 【关键修改 1：重算坐标】 ===
bar_width = 1.0
internal_gap = 0.6  # 组内间距（SR和GC之间的空隙）
group_gap = 2.5     # 组间间距（ALFRED和Habitat之间的空隙）

# ALFRED 坐标: [0, 1,  2+gap, 3+gap]
x_positions_alfred = np.array([0, 1, 2 + internal_gap, 3 + internal_gap])

# Habitat 坐标: 在 ALFRED 结束后开始
start_habitat = x_positions_alfred[-1] + group_gap
x_positions_habitat = np.array([start_habitat, start_habitat + 1, 
                                start_habitat + 2 + internal_gap, start_habitat + 3 + internal_gap])

# 分隔线位置：两个大组中间
separator_x = (x_positions_alfred[-1] + x_positions_habitat[0]) / 2

# =================================

# ALFRED柱子
for i in range(len(labels)):
    bar = ax.bar(x_positions_alfred[i], alfred_values[i], 
                width=bar_width,
                color=colors_refusal[i],
                alpha=alpha_refusal,
                edgecolor=edge_color,
                linewidth=edge_width,
                hatch=hatch_patterns[i],
                zorder=1)
    for bar_patch in bar:
        bar_patch.set_edgecolor(edge_color)
        bar_patch.set_hatch(hatch_patterns[i])
        bar_patch.set_linewidth(edge_width)
        bar_patch.set_alpha(alpha_refusal)
    ax.text(x_positions_alfred[i], alfred_values[i] + 1, f'{alfred_values[i]:.1f}',
            ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE, fontweight='bold', color='black')

# Habitat柱子
for i in range(len(labels)):
    bar = ax.bar(x_positions_habitat[i], habitat_values[i], 
                width=bar_width,
                color=colors_refusal[i],
                alpha=alpha_refusal,
                edgecolor=edge_color,
                linewidth=edge_width,
                hatch=hatch_patterns[i],
                zorder=1)
    for bar_patch in bar:
        bar_patch.set_edgecolor(edge_color)
        bar_patch.set_hatch(hatch_patterns[i])
        bar_patch.set_linewidth(edge_width)
        bar_patch.set_alpha(alpha_refusal)
    ax.text(x_positions_habitat[i], habitat_values[i] + 1, f'{habitat_values[i]:.1f}',
            ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE, fontweight='bold', color='black')

text_center_offset = 6.0 
arrow_color = '#1E88E5'

# --- 添加 "Experience Transfer" 文字 ---
# 计算新的中心点位置
alfred_center_x = (x_positions_alfred[0] + x_positions_alfred[-1]) / 2
habitat_center_x = (x_positions_habitat[0] + x_positions_habitat[-1]) / 2

bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="gray", ls="--", lw=1.5, alpha=0.9)

# ALFRED 文字 (居中显示)
ax.text(alfred_center_x, 70, "Experience Transfer", 
        ha='center', va='center', fontsize=15, fontweight='bold', color=arrow_color,
        bbox=bbox_props, zorder=21)

# Habitat 文字 (居中显示)
ax.text(habitat_center_x, 70, "Experience Transfer", 
        ha='center', va='center', fontsize=15, fontweight='bold', color=arrow_color,
        bbox=bbox_props, zorder=21)


# --- 连接曲线 ---
# 注意：由于使用了x_positions_alfred数组，箭头会自动跟随柱子移动，无需修改坐标逻辑

# ALFRED Arrows
con1 = ConnectionPatch(xyA=(x_positions_alfred[0], alfred_values[0] + text_center_offset), 
    xyB=(x_positions_alfred[1], alfred_values[1] + text_center_offset),
    coordsA="data", coordsB="data", axesA=ax, axesB=ax, arrowstyle="->", 
    shrinkA=3, shrinkB=3, mutation_scale=20, fc=arrow_color, ec=arrow_color, linewidth=2,
    connectionstyle="arc3,rad=-0.6", zorder=20)
ax.add_artist(con1)

con2 = ConnectionPatch(xyA=(x_positions_alfred[2], alfred_values[2] + text_center_offset), 
    xyB=(x_positions_alfred[3], alfred_values[3] + text_center_offset),
    coordsA="data", coordsB="data", axesA=ax, axesB=ax, arrowstyle="->", 
    shrinkA=3, shrinkB=3, mutation_scale=20, fc=arrow_color, ec=arrow_color, linewidth=2,
    connectionstyle="arc3,rad=-0.8", zorder=20)
ax.add_artist(con2)

# Habitat Arrows
con3 = ConnectionPatch(xyA=(x_positions_habitat[0], habitat_values[0] + text_center_offset), 
    xyB=(x_positions_habitat[1], habitat_values[1] + text_center_offset),
    coordsA="data", coordsB="data", axesA=ax, axesB=ax, arrowstyle="->", 
    shrinkA=3, shrinkB=3, mutation_scale=20, fc=arrow_color, ec=arrow_color, linewidth=2,
    connectionstyle="arc3,rad=-0.6", zorder=20)
ax.add_artist(con3)

con4 = ConnectionPatch(xyA=(x_positions_habitat[2], habitat_values[2] + text_center_offset), 
    xyB=(x_positions_habitat[3], habitat_values[3] + text_center_offset),
    coordsA="data", coordsB="data", axesA=ax, axesB=ax, arrowstyle="->", 
    shrinkA=3, shrinkB=3, mutation_scale=20, fc=arrow_color, ec=arrow_color, linewidth=2,
    connectionstyle="arc3,rad=-0.8", zorder=20)
ax.add_artist(con4)

# 分隔虚线
ax.axvline(x=separator_x, color=separator_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)

# 数据集标注 (使用计算出的中心点)
ax.text(alfred_center_x, -3, 'ALFRED', ha='center', va='center', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')
ax.text(habitat_center_x, -3, 'Habitat', ha='center', va='center', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')

# 坐标轴样式
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1)
    spine.set_alpha(0.9)

ax.set_axisbelow(True)
ax.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')

ax.set_xticks([])
# === 【关键修改 2：调整X轴范围】 ===
# 因为柱子位置整体变宽了，需要扩大显示范围
total_width = x_positions_habitat[-1] + 1
ax.set_xlim(-1, total_width + 0.5) 
ax.set_ylim(0, 100)
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, colors='#616161', width=1)
fig.patch.set_facecolor('white')

# 图例
legend_elements = []
for i in range(len(labels)):
    legend_elements.append(
        mpatches.Rectangle((0, 0), 1, 1, 
                           facecolor=colors_refusal[i], 
                           alpha=alpha_refusal,
                           edgecolor=edge_color, 
                           linewidth=2,
                           hatch=hatch_patterns[i],
                           label=f'{labels[i]}'))

ax.legend(handles=legend_elements,
          loc='upper center',
          # 稍微调整图例位置，确保居中
          bbox_to_anchor=(0.5, 1.01), 
          fontsize=LEGEND_FONTSIZE,
          frameon=True,
          framealpha=0.95,
          edgecolor='#BDBDBD',
          facecolor='white',
          shadow=False,
          ncol=2) # 这里如果不希望图例太宽，可以改 ncol=4 变成一行

plt.tight_layout()
plt.savefig('alfred_habitat_Transfer-3.5.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("图表已保存")
plt.show()

#### python bar_chart.py