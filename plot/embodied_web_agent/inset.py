import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pdf2image import convert_from_path
import tempfile
import os

# === 1. 基础设置 ===
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['hatch.linewidth'] = 1.2
mpl.rcParams['hatch.color'] = 'black'

# === 2. 数据准备 (Indoor Task Comparison) ===
categories = ['Embodied Acc', 'Web Acc', 'Completion Rate', 'Overall Acc']

### 4.1
react_values =      [32.17, 45.54, 21.88, 11.61]
worldmind_values =  [42.27, 80.36, 41.50, 24.11]

'''
### 3.5
react_values =      [29.99, 33.93, 17.02, 9.82]
worldmind_values =  [48.70, 45.54, 39.99, 20.54]
'''
# === 3. 样式设置 ===
styles = [
    {'color': '#f1c8c2', 'hatch': '/'},  # ReAct
    {'color': '#f3d0a9', 'hatch': '\\'}  # WorldMind
]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# 字体参数
GLOBAL_BASE_FONT_SIZE = 14
LABEL_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 4
TICK_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2 
LEGEND_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 1
ANNOTATION_FONTSIZE = GLOBAL_BASE_FONT_SIZE + 2
edge_color = 'black'
edge_width = 2.5
alpha_refusal = 1.0

# === 4. 绘制柱状图 ===
bar_width = 1.0
group_gap = 1.5 
indices = np.arange(len(categories))
group_centers = indices * (2 * bar_width + group_gap) 
x_react = group_centers - bar_width / 2
x_worldmind = group_centers + bar_width / 2

rects1 = ax.bar(x_react, react_values, width=bar_width,
                color=styles[0]['color'], edgecolor=edge_color, linewidth=edge_width,
                hatch=styles[0]['hatch'], alpha=alpha_refusal, zorder=1, label='ReAct')

rects2 = ax.bar(x_worldmind, worldmind_values, width=bar_width,
                color=styles[1]['color'], edgecolor=edge_color, linewidth=edge_width,
                hatch=styles[1]['hatch'], alpha=alpha_refusal, zorder=1, label='WorldMind')

# === 5. 添加数值标签 ===
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

# === 6. 坐标轴与图例 ===
ax.set_xticks(group_centers)
ax.set_xticklabels(categories, fontsize=TICK_FONTSIZE, fontweight='bold', color='#424242')
ax.set_xlim(min(x_react) - 1, max(x_worldmind) + 1)
ax.set_ylim(0, 95) 
ax.set_ylabel('Score (%)', fontsize=LABEL_FONTSIZE, fontweight='bold', color='#424242')
ax.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE, colors='#616161', width=1)
ax.tick_params(axis='x', which='major', length=0) 

for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1)
    spine.set_alpha(0.9)
ax.set_axisbelow(True)
fig.patch.set_facecolor('white')

legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=styles[0]['color'], edgecolor=edge_color, 
                       linewidth=2, hatch=styles[0]['hatch'], label='ReAct'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=styles[1]['color'], edgecolor=edge_color, 
                       linewidth=2, hatch=styles[1]['hatch'], label='WorldMind')
]

# 图例左上角
ax.legend(handles=legend_elements, loc='upper left', fontsize=LEGEND_FONTSIZE, 
          frameon=True, framealpha=0.95, edgecolor='#BDBDBD', facecolor='white', ncol=2)

# ==============================================================================
# === 7. 【核心修改】保持原比例插入 PDF ===
# ==============================================================================

pdf_path = '/mnt/20t/rbc/Embodied/EmbodiedBench/plot/embodied_web_agent/two_pie_charts_4.1.pdf'

try:
    if os.path.exists(pdf_path):
        # 1. 转换 PDF (dpi=300 保证清晰度)
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        
        if pages:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pages[0].save(tmp_file.name, 'PNG')
                
                # 2. 创建右上角子图区域
                # 这里的 width 和 height 定义了图片可以放置的“最大容器区域”
                # 您可以适当把这个区域设大一点（比如50%），图片会自动缩放以适应这个区域，且保持比例
                inset_ax = inset_axes(ax, width="35%", height="35%", 
                                    bbox_to_anchor=(0.00, 0.05, 1, 1), # (x, y, w, h)
                                    bbox_transform=ax.transAxes, 
                                    loc='upper right')
                
                # 3. 读取图片
                img = mpimg.imread(tmp_file.name)
                
                # 4. 【关键修改】显示图片时保持比例
                # 去掉了 aspect='auto'，默认就是 aspect='equal' (保持比例)
                # 图片会根据最长边自动缩放以适应 inset_axes 的框，不会变形
                inset_ax.imshow(img) 
                
                # 5. 隐藏子图的坐标轴和边框
                # 这样即使图片和容器比例不一致，留下的白边也是不可见的
                inset_ax.axis('off')
                
                os.unlink(tmp_file.name)
    else:
        print(f"警告: PDF文件未找到: {pdf_path}")
        # 如果找不到文件，显示占位符
        inset_ax = inset_axes(ax, width="30%", height="30%", loc='upper right')
        inset_ax.text(0.5, 0.5, 'PDF Not Found', ha='center', va='center', color='red')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        
except Exception as e:
    print(f"插入PDF图片时出错: {e}")
    import traceback
    traceback.print_exc()

# ==============================================================================

plt.tight_layout()
plt.savefig('indoor_task_4.1_inset.pdf', 
            format='pdf', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("图表已保存为 'indoor_task_4.1_inset.pdf'")
plt.show()

### python inset.py
