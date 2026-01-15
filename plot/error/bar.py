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

# === 2. 数据准备 (更新为错误类型数据) ===
categories = ['Invalid\nActions', 'Timeout', 'Wrong\nTermination']

# 对应的数据值
# Invalid Actions: 105 vs 67
# Timeout: 4 vs 30
# Wrong Termination: 32 vs 30
'''

### 3.5-habitat
react_values =     [105, 4, 32]
worldmind_values = [67, 30, 30]



### 4.1-habitat
react_values =     [107, 5, 34]
worldmind_values = [69, 18, 35]




### 3.5-alfred
react_values =     [35, 63, 40]
worldmind_values = [52, 49, 22]

## 4.1-alfred
react_values =     [47, 51, 46]
worldmind_values = [49, 59, 19]

'''
### 3.5-habitat
react_values =     [105, 4, 32]
worldmind_values = [67, 30, 30]



# === 3. 样式设置 ===
# 保持你代码中的配色设置
styles = [
    {'color': '#f1c8c2', 'hatch': '/'},   # ReAct (粉色)
    {'color': '#f3d0a9', 'hatch': '\\'}   # WorldMind (米黄色)
]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=100)

# 字体参数
GLOBAL_BASE_FONT_SIZE = 21                   # 全局基础字体大小
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 9   # Y轴标签字体大小（ax.set_ylabel）
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 6    # 坐标轴刻度字体大小（ax.set_xticklabels, ax.tick_params）
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 7  # 图例字体大小（ax.legend）
ANNOTATION_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 10  # 柱子顶部数值标签字体大小（autolabel函数）
edge_color = 'black'                         # 柱子边框颜色
edge_width = 3.5                             # 柱子边框宽度
alpha_refusal = 1.0                          # 柱子透明度

# === 4. 计算坐标位置 ===
bar_width = 2.0
group_gap = 1.2 

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
        
        ax.text(rect.get_x() + rect.get_width() / 2., height + 1, # +1 稍微抬高一点防止压线
                f'{int(height)}', # 错误数通常是整数，这里去掉了小数位
                ha='center', va='bottom', 
                fontsize=ANNOTATION_FONTSIZE, fontweight='bold', color='black')

autolabel(rects1, styles[0]['hatch'])
autolabel(rects2, styles[1]['hatch'])

# === 7. 坐标轴与图例 ===
ax.set_xticks(group_centers)
ax.set_xticklabels(categories, fontsize=TICK_FONTSIZE, fontweight='bold', color='#424242')

# 动态调整X轴范围
ax.set_xlim(min(x_react) - 1.5, max(x_worldmind) + 1.5)

# Y轴设置
# 【关键修改】因为Invalid Actions高达105，需要拉高上限
ax.set_ylim(0, 160) 
# 【关键修改】标签改为 Error Count
ax.set_ylabel('Error Count', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')
ax.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE+6, colors='black', width=1)
ax.tick_params(axis='x', which='major', length=0) 

# 边框
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1.2)
    spine.set_alpha(1.0)
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
          loc='upper right',
          bbox_to_anchor=(1.0, 1.01), # 改为居中，因为只有3个组
          fontsize=LEGEND_FONTSIZE,
          frameon=True,
          framealpha=0.95,
          edgecolor='#BDBDBD',
          facecolor='white',
          shadow=False,
          ncol=2)

plt.tight_layout()
plt.savefig('error_analysis_3.5-habitat.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("错误分析图表已保存为 'error_analysis_3.5-habitat.pdf'")
plt.show()


### python bar.py