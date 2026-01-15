import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import numpy as np

# === 1. 基础设置 ===
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['hatch.color'] = 'black'

# === 2. 数据准备 ===
all_labels = [
    'Max Steps Reached', 
    'Step Execution Error', 
    'Repeated Web Actions', 
    'Repeated Embodied Actions'
]
all_colors = ['#b3d9ff', '#f1c8c2', '#bfd3c2', '#f3d0a9']
all_hatches = ['.', '/', '\\', '+']

'''
### 3.5
sizes_1 = [68, 22, 17, 5]
sizes_2 = [93, 12, 0, 7]

'''

### 4.1
sizes_1 = [73, 27, 12, 0]
sizes_2 = [96, 8, 0, 8]

# --- Chart 1: 过滤掉值为0的项 ---
filtered_sizes_1 = []
filtered_colors_1 = []
filtered_hatches_1 = []
filtered_explode_1 = []

for i, size in enumerate(sizes_1):
    if size > 0:
        filtered_sizes_1.append(size)
        filtered_colors_1.append(all_colors[i])
        filtered_hatches_1.append(all_hatches[i])
        # Step Execution Error (索引1) 需要突出显示
        if i == 1:
            filtered_explode_1.append(0.1)
        else:
            filtered_explode_1.append(0)

# --- Chart 2: 过滤掉值为0的项 ---
filtered_sizes_2 = []
filtered_colors_2 = []
filtered_hatches_2 = []
filtered_explode_2 = []

for i, size in enumerate(sizes_2):
    if size > 0:
        filtered_sizes_2.append(size)
        filtered_colors_2.append(all_colors[i])
        filtered_hatches_2.append(all_hatches[i])
        # Step Execution Error (索引1) 需要突出显示
        if i == 1:
            filtered_explode_2.append(0.1)
        else:
            filtered_explode_2.append(0)

# === 3. 创建画布 ===
fig, ax = plt.subplots(1, 2, figsize=(14, 7.5), dpi=150) 
plt.subplots_adjust(top=0.80, wspace=0.1)

# === 4. 参数控制区域 ===
legend_col_spacing = 3.0
legend_row_spacing = 0.2
legend_y_pos = 0.87
title_y_pos = 0.03

# === 5. 绘图函数 ===
def draw_pie(axis, sizes, colors, hatches, explode, title_text):
    wedges, texts, autotexts = axis.pie(sizes, 
                                      explode=explode, 
                                      labels=None, 
                                      colors=colors, 
                                      autopct='%1.1f%%',
                                      pctdistance=0.72,
                                      shadow=False, 
                                      startangle=140, 
                                      wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'antialiased': True},
                                      textprops={'fontsize': 13, 'fontfamily': 'serif'}
                                     )
    for i, wedge in enumerate(wedges):
        wedge.set_hatch(hatches[i])
        wedge.set_edgecolor('black')

    for t in autotexts:
        t.set_fontsize(14)
        t.set_color('black')
        t.set_fontweight('bold')
        t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    
    axis.set_title(title_text, y=title_y_pos, fontsize=18, fontweight='bold')

# === 6. 绘制两个图 ===
draw_pie(ax[0], filtered_sizes_1, filtered_colors_1, filtered_hatches_1, filtered_explode_1, "Baseline (ReAct)")
draw_pie(ax[1], filtered_sizes_2, filtered_colors_2, filtered_hatches_2, filtered_explode_2, "Ours (WorldMind)")

# === 7. 添加统一图例 (显示所有4个类别) ===
legend_handles = []
for i in range(len(all_labels)):
    patch = mpatches.Patch(
        facecolor=all_colors[i], 
        edgecolor='black', 
        label=all_labels[i],
        hatch=all_hatches[i]
    )
    legend_handles.append(patch)

fig.legend(handles=legend_handles, 
           loc='upper center', 
           bbox_to_anchor=(0.5, legend_y_pos),
           ncol=2,
           fontsize=15, 
           frameon=False,
           columnspacing=legend_col_spacing,
           labelspacing=legend_row_spacing,
           handlelength=2, 
           handleheight=1.5
          )

# === 8. 添加直箭头和文字 ===
xy_a = (0.98, 0.5)  
xy_b = (0.02, 0.5) 

con = ConnectionPatch(
    xyA=xy_a, coordsA=ax[0].transAxes,
    xyB=xy_b, coordsB=ax[1].transAxes,
    arrowstyle="-|>", 
    shrinkA=0, shrinkB=0,
    mutation_scale=50, 
    color="#BFBFBF",
    linewidth=3,
    zorder=10
)
fig.add_artist(con)

plt.savefig('two_pie_charts_4.1.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("图表已保存为 'two_pie_charts_4.1.pdf'")
plt.show()

### python pie.py