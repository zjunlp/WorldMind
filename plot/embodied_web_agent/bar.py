import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

# === 1. 基础设置 ===
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['hatch.linewidth'] = 1.2
mpl.rcParams['hatch.color'] = 'black'

# === 2. 数据准备 (已按新顺序调整) ===
# 新顺序: Embodied Acc -> Web Acc -> Completion Rate -> Overall Acc
categories = ['Embodied Acc', 'Web Acc', 'Completion Rate', 'Overall Acc']

# 对应的数据值
# Embodied Acc: 29.99, 48.70
# Web Acc: 33.93, 45.54
# Completion Rate: 17.02, 39.99
# Overall Acc: 9.82, 20.54
react_values =     [29.99, 33.93, 17.02, 9.82]
worldmind_values = [48.70, 45.54, 39.99, 20.54]

# === 3. 样式设置 (颜色修改) ===
# ReAct 保持粉色，WorldMind 改为淡蓝色 (#b3d9ff) 搭配更好看
styles = [
    {'color': '#f1c8c2', 'hatch': '/'},  # ReAct (粉色)
    {'color': '#f3d0a9', 'hatch': '\\'}   # WorldMind (淡蓝色 - 新颜色)
]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)

# 字体参数
GLOBAL_BASE_FONT_SIZE = 14
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2 
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 1
ANNOTATION_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2
edge_color = 'black'
edge_width = 2.5
alpha_refusal = 1.0

# === 4. 计算坐标位置 ===
bar_width = 1.0
group_gap = 1.5 

indices = np.arange(len(categories))
group_centers = indices * (2 * bar_width + group_gap) 

x_react = group_centers - bar_width / 2
x_worldmind = group_centers + bar_width / 2

# === 5. 绘制柱状图 ===
# ReAct
rects1 = ax.bar(x_react, react_values, width=bar_width,
                color=styles[0]['color'], edgecolor=edge_color, linewidth=edge_width,
                hatch=styles[0]['hatch'], alpha=alpha_refusal, zorder=1, label='ReAct')

# WorldMind
rects2 = ax.bar(x_worldmind, worldmind_values, width=bar_width,
                color=styles[1]['color'], edgecolor=edge_color, linewidth=edge_width,
                hatch=styles[1]['hatch'], alpha=alpha_refusal, zorder=1, label='WorldMind')

# === 6. 添加数值标签 ===
def autolabel(rects, hatch_pattern):
    for rect in rects:
        height = rect.get_height()
        rect.set_edgecolor(edge_color)
        rect.set_linewidth(edge_width)
        rect.set_hatch(hatch_pattern)
        
        ax.text(rect.get_x() + rect.get_width() / 2., height + 0.5,
                f'{height:.2f}', 
                ha='center', va='bottom', 
                fontsize=ANNOTATION_FONTSIZE, fontweight='bold', color='black')

autolabel(rects1, styles[0]['hatch'])
autolabel(rects2, styles[1]['hatch'])

# === 7. 坐标轴与图例 ===
ax.set_xticks(group_centers)
ax.set_xticklabels(categories, fontsize=TICK_FONTSIZE, fontweight='bold', color='#424242')

# 动态调整X轴范围
ax.set_xlim(min(x_react) - 1, max(x_worldmind) + 1)

# Y轴设置
ax.set_ylim(0, 60) 
ax.set_ylabel('Score (%)', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')
ax.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE, colors='#616161', width=1)
ax.tick_params(axis='x', which='major', length=0) 

# 边框
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1)
    spine.set_alpha(0.9)
ax.set_axisbelow(True)
fig.patch.set_facecolor('white')

# 图例
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=styles[0]['color'], edgecolor=edge_color, 
                       linewidth=2, hatch=styles[0]['hatch'], label='ReAct'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=styles[1]['color'], edgecolor=edge_color, 
                       linewidth=2, hatch=styles[1]['hatch'], label='WorldMind')
]

ax.legend(handles=legend_elements,
          loc='upper center',
          bbox_to_anchor=(0.25, 1.01),
          fontsize=LEGEND_FONTSIZE,
          frameon=True,
          framealpha=0.95,
          edgecolor='#BDBDBD',
          facecolor='white',
          shadow=False,
          ncol=2)

plt.tight_layout()
plt.savefig('indoor_task_comparison.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("图表已保存为 'indoor_task_comparison.pdf'")
plt.show()

### python bar.py