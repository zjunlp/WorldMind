import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False

# ==========================================
# 全局参数设置
# ==========================================
MARKER_SIZE_SQUARE = 15
MARKER_SIZE_STAR = 21
VALUE_FONT_SIZE = 24
GLOBAL_BASE_FONT_SIZE = 18
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE 
# ==========================================

# 数据保持不变...
# GPT-3.5-turbo
gpt35_alfred_sr = [44.4, 48.8]
gpt35_alfred_gc = [50.4, 57.0]
gpt35_habitat_sr = [43.6, 46.4]
gpt35_habitat_gc = [50.4, 54.2]

# GPT-4.1-mini
gpt4_alfred_sr = [41.2, 46.4]
gpt4_alfred_gc = [47.5, 51.9]
gpt4_habitat_sr = [41.6, 54.2]
gpt4_habitat_gc = [47.4, 51.9]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 颜色设置
sr_color = '#E57373'
gc_color = '#64B5F6'
separator_color = '#CCCCCC'
grid_color = '#E0E0E0' # 新增网格颜色：浅灰色

# X轴位置设置
gpt35_alfred_x = [0, 1]
gpt35_habitat_x = [2, 3]
gpt4_alfred_x = [5, 6]
gpt4_habitat_x = [7, 8]

separator_x = 4

# ==========================================
# 修改部分：移除背景填充，添加网格
# ==========================================

# 1. 移除背景填充 (注释掉或删除这两行)
# ax.axvspan(-0.5, separator_x, facecolor='#ebe7f5', alpha=0.25, zorder=0)
# ax.axvspan(separator_x, 8.5, facecolor='#FFF9C4', alpha=0.25, zorder=0)

# 2. 添加浅灰色背景网格
# axis='y' 表示只显示水平网格线，辅助阅读 Y 轴数值
# linestyle='--' 设置为虚线，视觉上更轻盈
# zorder=0 确保网格在数据下方
ax.grid(True, which='major', axis='both', linestyle='--', linewidth=1, color=grid_color, alpha=0.8, zorder=0)

# ==========================================


# ==========================================
# 绘图部分 (GPT-3.5)
# ==========================================
# ALFRED
ax.plot(gpt35_alfred_x, gpt35_alfred_sr, linewidth=2.5, color=sr_color, zorder=2)
ax.plot(gpt35_alfred_x[0], gpt35_alfred_sr[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=sr_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt35_alfred_x[1], gpt35_alfred_sr[1], marker='*', markersize=MARKER_SIZE_STAR, color=sr_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

ax.plot(gpt35_alfred_x, gpt35_alfred_gc, linewidth=2.5, color=gc_color, zorder=2)
ax.plot(gpt35_alfred_x[0], gpt35_alfred_gc[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=gc_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt35_alfred_x[1], gpt35_alfred_gc[1], marker='*', markersize=MARKER_SIZE_STAR, color=gc_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

# 数值标注
ax.text(gpt35_alfred_x[0], gpt35_alfred_sr[0] + 1.5, f'{gpt35_alfred_sr[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_alfred_x[1], gpt35_alfred_sr[1] + 1.5, f'{gpt35_alfred_sr[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_alfred_x[0], gpt35_alfred_gc[0] + 1.5, f'{gpt35_alfred_gc[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_alfred_x[1], gpt35_alfred_gc[1] + 1.5, f'{gpt35_alfred_gc[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')

# Habitat
ax.plot(gpt35_habitat_x, gpt35_habitat_sr, linewidth=2.5, color=sr_color, zorder=2)
ax.plot(gpt35_habitat_x[0], gpt35_habitat_sr[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=sr_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt35_habitat_x[1], gpt35_habitat_sr[1], marker='*', markersize=MARKER_SIZE_STAR, color=sr_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

ax.plot(gpt35_habitat_x, gpt35_habitat_gc, linewidth=2.5, color=gc_color, zorder=2)
ax.plot(gpt35_habitat_x[0], gpt35_habitat_gc[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=gc_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt35_habitat_x[1], gpt35_habitat_gc[1], marker='*', markersize=MARKER_SIZE_STAR, color=gc_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

# 数值标注
ax.text(gpt35_habitat_x[0], gpt35_habitat_sr[0] + 1.5, f'{gpt35_habitat_sr[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_habitat_x[1], gpt35_habitat_sr[1] + 1.5, f'{gpt35_habitat_sr[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_habitat_x[0], gpt35_habitat_gc[0] + 1.5, f'{gpt35_habitat_gc[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt35_habitat_x[1], gpt35_habitat_gc[1] + 1.5, f'{gpt35_habitat_gc[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')

# ==========================================
# 绘图部分 (GPT-4.1)
# ==========================================
# ALFRED
ax.plot(gpt4_alfred_x, gpt4_alfred_sr, linewidth=2.5, color=sr_color, zorder=2)
ax.plot(gpt4_alfred_x[0], gpt4_alfred_sr[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=sr_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt4_alfred_x[1], gpt4_alfred_sr[1], marker='*', markersize=MARKER_SIZE_STAR, color=sr_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

ax.plot(gpt4_alfred_x, gpt4_alfred_gc, linewidth=2.5, color=gc_color, zorder=2)
ax.plot(gpt4_alfred_x[0], gpt4_alfred_gc[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=gc_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt4_alfred_x[1], gpt4_alfred_gc[1], marker='*', markersize=MARKER_SIZE_STAR, color=gc_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

# 数值标注
ax.text(gpt4_alfred_x[0], gpt4_alfred_sr[0] + 1.5, f'{gpt4_alfred_sr[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt4_alfred_x[1], gpt4_alfred_sr[1] + 1.5, f'{gpt4_alfred_sr[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt4_alfred_x[0], gpt4_alfred_gc[0] + 1.5, f'{gpt4_alfred_gc[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt4_alfred_x[1], gpt4_alfred_gc[1] + 1.5, f'{gpt4_alfred_gc[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')

# Habitat
ax.plot(gpt4_habitat_x, gpt4_habitat_sr, linewidth=2.5, color=sr_color, zorder=2)
ax.plot(gpt4_habitat_x[0], gpt4_habitat_sr[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=sr_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt4_habitat_x[1], gpt4_habitat_sr[1], marker='*', markersize=MARKER_SIZE_STAR, color=sr_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

ax.plot(gpt4_habitat_x, gpt4_habitat_gc, linewidth=2.5, color=gc_color, zorder=2)
ax.plot(gpt4_habitat_x[0], gpt4_habitat_gc[0], marker='s', markersize=MARKER_SIZE_SQUARE, color=gc_color, markeredgecolor='black', markeredgewidth=1.5, zorder=3)
ax.plot(gpt4_habitat_x[1], gpt4_habitat_gc[1], marker='*', markersize=MARKER_SIZE_STAR, color=gc_color, markeredgecolor='black', markeredgewidth=1, zorder=3)

# 数值标注
ax.text(gpt4_habitat_x[0], gpt4_habitat_sr[0] + 1.5, f'{gpt4_habitat_sr[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt4_habitat_x[1], gpt4_habitat_sr[1] + 1.5, f'{gpt4_habitat_sr[1]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')
ax.text(gpt4_habitat_x[0], gpt4_habitat_gc[0] + 1.5, f'{gpt4_habitat_gc[0]:.1f}', ha='center', va='bottom', fontsize=VALUE_FONT_SIZE, fontweight='bold')

# 特殊处理的点：数值在下方
ax.text(gpt4_habitat_x[1], gpt4_habitat_gc[1] - 2.0, f'{gpt4_habitat_gc[1]:.1f}',
        ha='center', va='top', fontsize=VALUE_FONT_SIZE, fontweight='bold')

# 分隔虚线
ax.axvline(x=separator_x, color=separator_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)

# GPT 模型标签
MODEL_LABEL_SIZE = 18
LABEL_Y_POS = 65
box_props = dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#B0B0B0', linestyle='--', linewidth=1.5, alpha=0.9)
ax.text(1.5, LABEL_Y_POS, 'GPT-4.1-mini-EXP\n↓\nGPT-3.5-turbo', 
        ha='center', va='center', fontsize=MODEL_LABEL_SIZE+2, fontstyle='italic', color='#757575', bbox=box_props, zorder=10)
ax.text(6.5, LABEL_Y_POS, 'GPT-3.5-turbo-EXP\n↓\nGPT-4.1-mini', 
        ha='center', va='center', fontsize=MODEL_LABEL_SIZE+2, fontstyle='italic', color='#757575', bbox=box_props, zorder=10)

# 坐标轴设置
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1)
    spine.set_alpha(0.9)

# 确保网格线在数据下方
ax.set_axisbelow(True)
ax.set_ylabel('Success Rate (%)', fontsize=LABEL_FONTSIZE+4, fontweight='bold', color='#424242')

x_tick_positions = [0.5, 2.5, 5.5, 7.5]
x_tick_labels = ['ALFRED', 'Habitat', 'ALFRED', 'Habitat']
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(x_tick_labels, fontsize=TICK_FONTSIZE+1, color='#424242', fontweight='bold')

ax.set_xlim(-0.5, 8.5)
ax.set_ylim(40, 72)

ax.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE+4, colors='#616161', width=1)
ax.tick_params(axis='x', which='major', length=0)
fig.patch.set_facecolor('white')
# ax.set_facecolor('none') # 不需要设置透明了，使用默认白色即可

# 图例
legend_elements = [
    Line2D([0], [0], color=sr_color, linewidth=2.5, label='Metric: SR'),
    Line2D([0], [0], color=gc_color, linewidth=2.5, label='Metric: GC'),
    Line2D([0], [0], marker='s', markersize=MARKER_SIZE_SQUARE, color='gray', linestyle='None',
           markeredgecolor='black', markeredgewidth=1.5, label='ReAct'),
    Line2D([0], [0], marker='*', markersize=MARKER_SIZE_STAR, color='gray', linestyle='None',
           markeredgecolor='black', markeredgewidth=1, label='WorldMind')
]

ax.legend(
    handles=legend_elements,
    loc='upper center',
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    framealpha=0.95,
    edgecolor='#BDBDBD',
    facecolor='white',
    shadow=False,
    ncol=4,
    bbox_to_anchor=(0.5, 1.02),
    columnspacing=0.6,
    handletextpad=0.0
)

plt.tight_layout()
plt.savefig('transfer_comparison_grid.pdf', format='pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("带网格的图表已保存为 'transfer_comparison_grid.pdf'")
plt.show()

### python line1.py