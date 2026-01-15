import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False

steps = ['Step-100', 'Step-150', 'Step-200', 'Step-250', 'Step-300']
# 更新为新的incorrect数据
simpleqa_incorrect_values = [74.0, 64.0, 75.33, 83.67, 83.0]
chinese_incorrect_values = [74.33, 65.66, 73.0, 68.33, 70.33]

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=100)

# 新配色方案
color_incorrect = '#FFD700'      # 黄色
color_incorrect_chinese = '#32CD32'  # 绿色
separator_color = '#CCCCCC'

GLOBAL_BASE_FONT_SIZE = 14
TITLE_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 3
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 8
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 1
ANNOTATION_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2

x_positions_simpleqa = np.array([-0.5, 0.75, 2, 3.25, 4.5])
x_positions_chinese = np.array([5.5, 6.75, 8, 9.25, 10.5])
separator_x = 5

# SimpleQA填充区域 - 从X轴到折线
ax.fill_between(x_positions_simpleqa, 60, simpleqa_incorrect_values, 
                color=color_incorrect, alpha=0.2, zorder=1)

# ChineseSimpleQA填充区域 - 从X轴到折线
ax.fill_between(x_positions_chinese, 60, chinese_incorrect_values, 
                color=color_incorrect_chinese, alpha=0.2, zorder=1)

# SimpleQA incorrect菱形及折线
ax.plot(x_positions_simpleqa, simpleqa_incorrect_values, color=color_incorrect, linewidth=2.5, marker='D', markersize=10, zorder=5)
for i in range(len(steps)):
    ax.scatter(x_positions_simpleqa[i], simpleqa_incorrect_values[i],
              s=180,
              color=color_incorrect,
              alpha=0.9,
              edgecolor='white',
              linewidth=2,
              marker='D',
              zorder=6)
    # 数字不用方框
    ax.text(x_positions_simpleqa[i], simpleqa_incorrect_values[i] + 2, f'{simpleqa_incorrect_values[i]:.1f}', 
            ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE,
            fontweight='bold', color=color_incorrect)

# ChineseSimpleQA incorrect菱形及折线
ax.plot(x_positions_chinese, chinese_incorrect_values, color=color_incorrect_chinese, linewidth=2.5, marker='D', markersize=10, zorder=5)
for i in range(len(steps)):
    ax.scatter(x_positions_chinese[i], chinese_incorrect_values[i],
              s=180,
              color=color_incorrect_chinese,
              alpha=0.9,
              edgecolor='white',
              linewidth=2,
              marker='D',
              zorder=6)
    # 数字不用方框
    ax.text(x_positions_chinese[i], chinese_incorrect_values[i] + 2, f'{chinese_incorrect_values[i]:.1f}', 
            ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE,
            fontweight='bold', color=color_incorrect_chinese)

# 分隔虚线
ax.axvline(x=separator_x, color=separator_color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)

# 数据集标注
ax.text(2, 61, 'SimpleQA', ha='center', va='center', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')
ax.text(8, 61, 'ChineseSimpleQA', ha='center', va='center', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')

# 坐标轴样式
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1)
    spine.set_alpha(0.6)

ax.set_axisbelow(True)
ax.set_ylabel('Incorrect Rate (%)', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')

# 不设置标题

# 设置X轴刻度和标签 - 调整字体、位置和间距
all_x_positions = list(x_positions_simpleqa) + list(x_positions_chinese)
step_labels = ['0', '100','150', '200', '250',  '0', '100', '150', '200', '250']

ax.set_xticks(all_x_positions)
ax.set_xticklabels(step_labels, 
                   fontsize=TICK_FONTSIZE-4,  # 缩小字体
                   fontweight='normal',        # 调整字体粗细
                   color='#888888')           # 调整字体颜色

ax.set_xlim(-1, 11)
ax.set_ylim(60, 90)  # 调整Y轴范围适应新数据
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, colors='#616161', width=1)
fig.patch.set_facecolor('white')

# 图例 - 缩小长度
legend_elements = []
legend_elements.append(
    plt.Line2D([0], [0], marker='D', color=color_incorrect, markerfacecolor=color_incorrect, 
               markersize=8, markeredgecolor='white', markeredgewidth=1,
               label='SimpleQA'))
legend_elements.append(
    plt.Line2D([0], [0], marker='D', color=color_incorrect_chinese, markerfacecolor=color_incorrect_chinese, 
               markersize=8, markeredgecolor='white', markeredgewidth=1,
               label='ChineseSimpleQA'))

ax.legend(handles=legend_elements,
          loc='upper right',
          bbox_to_anchor=(1.02, 1.02),
          fontsize=LEGEND_FONTSIZE,
          frameon=True,
          framealpha=0.95,
          edgecolor='#BDBDBD',
          facecolor='white',
          shadow=False,
          handlelength=1.5,  # 缩小图例符号长度
          handletextpad=0.5)  # 缩小符号和文字间距

plt.tight_layout()
plt.savefig('incorrect_rates_analysis.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("Incorrect图表已保存为 'incorrect_rates_analysis.pdf'")
plt.show()


### python line.py