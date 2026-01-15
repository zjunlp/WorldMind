import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
CONFIG = {
    # --- 字体大小设置 ---
    'font_size_value': 21,        # 柱子上数值标签的字体大小
    'font_size_tick': 23,         # X/Y轴刻度标签的字体大小
    'font_size_label': 23,        # Y轴标题(Label)的字体大小
    'font_size_model_title': 20,  # 顶部模型名称(如GPT-3.5)的字体大小
    'font_size_legend': 17.5,       # 图例字体大小

    # --- 布局与距离控制 ---
    'dist_react_wm': 3.0,         # ReAct组与WorldMind组中心的距离
    'dist_between_models': 2.5,   # 左右大模型区域之间的间距
    'bar_width': 0.6,             # 柱子的宽度
    
    # 【参数说明】现在这个参数控制的是"柱子边缘"到"画布边缘"的绝对物理距离
    # 设为 0.6 大约等于一个柱子的宽度，看着比较舒服
    'x_axis_margin': 0.2,         
    
    # --- 标签与位置参数 ---
    'model_label_y': 108,         
    'arrow_y_pos': 80,            
    'legend_y_pos': 1.025,         
    'ylim_max': 135,              
    
    # --- 样式与颜色 ---
    'colors': ['#f1c8c2', '#fee5d4', '#f3d0a9', '#bfd3c2'],
    'hatch_patterns': ['/', '+', '\\', '.'], 
    'edge_color': 'black',
    'edge_width': 1.5,            
    'separator_color': '#CCCCCC',

    # 【新增配置】箭头的颜色
    'arrow_color': '#E57373' # 一种柔和的红色
}

# ==========================================
# 2. 基础样式设置
# ==========================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['hatch.linewidth'] = 1.0
mpl.rcParams['hatch.color'] = CONFIG['edge_color']

# ==========================================
# 3. 数据准备
# ==========================================
group_labels_clean = ['ReAct', 'WorldMind', 'ReAct', 'WorldMind']
metric_labels = ['Max Steps Reached', 'Step Execution Error', 'Repeated Web Actions', 'Repeated Embodied Actions']

data = np.array([
    [68, 22, 17,  5],  # GPT-3.5 ReAct (4根)
    [93, 12,  0,  7],  # GPT-3.5 WorldMind
    [73, 27, 12,  0],  # GPT-4.1 ReAct
    [96,  8,  0,  8]   # GPT-4.1 WorldMind (3根)
])

# ==========================================
# 4. 坐标计算逻辑
# ==========================================
center_1 = 1.5 
center_2 = center_1 + CONFIG['dist_react_wm']
center_3 = center_2 + CONFIG['dist_between_models']
center_4 = center_3 + CONFIG['dist_react_wm']

group_centers = np.array([center_1, center_2, center_3, center_4])
separator_pos = (center_2 + center_3) / 2

# ==========================================
# 5. 绘图核心逻辑
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 变量用于记录由于动态去0后，实际绘图的最左和最右边界
actual_visual_left_edge = 0
actual_visual_right_edge = 0

# --- 循环绘制柱子 ---
for group_idx, center_x in enumerate(group_centers):
    row_data = data[group_idx]
    
    # 找出非0数据
    non_zero_indices = np.where(row_data > 0)[0]
    num_active_bars = len(non_zero_indices)
    
    if num_active_bars == 0:
        continue

    # 计算该组的总物理宽度
    total_cluster_width = num_active_bars * CONFIG['bar_width']
    
    # 计算第一根柱子中心的起始位置
    start_x = center_x - (total_cluster_width / 2) + (CONFIG['bar_width'] / 2)
    
    # 【关键修复】记录物理边缘用于设置X轴范围
    # 如果是第一组 (最左边)
    if group_idx == 0:
        # 左边缘 = 第一组中心 - 半个总宽
        actual_visual_left_edge = center_x - (total_cluster_width / 2)
    # 如果是最后一组 (最右边)
    if group_idx == 3:
        # 右边缘 = 最后一组中心 + 半个总宽
        actual_visual_right_edge = center_x + (total_cluster_width / 2)

    for i, metric_idx in enumerate(non_zero_indices):
        val = row_data[metric_idx]
        current_x = start_x + i * CONFIG['bar_width']
        
        # 绘制
        bar_container = ax.bar(current_x, val, 
                               width=CONFIG['bar_width'],
                               color=CONFIG['colors'][metric_idx],
                               alpha=1.0,
                               edgecolor=CONFIG['edge_color'],
                               linewidth=CONFIG['edge_width'],
                               hatch=CONFIG['hatch_patterns'][metric_idx],
                               zorder=2)
        
        for bar_patch in bar_container:
            bar_patch.set_edgecolor(CONFIG['edge_color'])
            bar_patch.set_hatch(CONFIG['hatch_patterns'][metric_idx])
            bar_patch.set_linewidth(CONFIG['edge_width'])
            
        ax.text(current_x, val + 2, f'{int(val)}',
                ha='center', va='bottom', 
                fontsize=CONFIG['font_size_value'], 
                fontweight='bold', color='black', zorder=3)

# ==========================================
# 6. 装饰与标注
# ==========================================

# (1) 绘制中央分割虚线
ax.axvline(x=separator_pos, color=CONFIG['separator_color'], linestyle='--', linewidth=2.0, alpha=0.8, zorder=1)

# (2) 添加顶部模型标签
box_props = dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#B0B0B0', linestyle='--', linewidth=1.5, alpha=0.9)

left_region_center = (center_1 + center_2) / 2
ax.text(left_region_center-1, CONFIG['model_label_y']-10, 'GPT-3.5-turbo', 
        ha='center', va='center', 
        fontsize=CONFIG['font_size_model_title'], 
        fontstyle='italic', color='#424242', 
        bbox=box_props, zorder=10)

right_region_center = (center_3 + center_4) / 2
ax.text(right_region_center-1, CONFIG['model_label_y']-10, 'GPT-4.1-mini', 
        ha='center', va='center', 
        fontsize=CONFIG['font_size_model_title'], 
        fontstyle='italic', color='#424242', 
        bbox=box_props, zorder=10)

# ==========================================
# (3) 添加箭头 【关键修改部分】
# ==========================================
# 使用新的红色配置
arrow_style = dict(facecolor=CONFIG['arrow_color'], edgecolor=CONFIG['arrow_color'], 
                   alpha=0.9, width=2.0, headwidth=10, headlength=10)

# 左侧箭头 (GPT-3.5)
# xy是终点，xytext是起点
# 修改：将终点 center_2 - 0.6 改为 center_2 - 1.2，使其更早结束
ax.annotate('', 
            xy=(center_2 - 1.2, CONFIG['arrow_y_pos']),      # 终点 (WorldMind左侧)
            xytext=(center_1 + 0.4, CONFIG['arrow_y_pos']),  # 起点 (ReAct右侧)
            arrowprops=arrow_style, zorder=5)

# 右侧箭头 (GPT-4.1)
# 修改：将终点 center_4 - 0.6 改为 center_4 - 1.2，使其更早结束
ax.annotate('', 
            xy=(center_4 - 1.2, CONFIG['arrow_y_pos']),      # 终点 (WorldMind左侧)
            xytext=(center_3 + 0.4, CONFIG['arrow_y_pos']),  # 起点 (ReAct右侧)
            arrowprops=arrow_style, zorder=5)
# ==========================================


# (4) 坐标轴设置
ax.set_ylabel('Number of Occurrences', fontsize=CONFIG['font_size_label'], fontweight='bold', color='#424242')
ax.set_xticks(group_centers)
ax.set_xticklabels(group_labels_clean, fontsize=CONFIG['font_size_tick'], fontweight='bold', color='#424242')

ax.set_ylim(0, CONFIG['ylim_max'])

# 【关键修改】基于实际的物理边缘设置X轴范围
# 这样无论左右两边有多少根柱子，留白都是完全对称的
ax.set_xlim(actual_visual_left_edge - CONFIG['x_axis_margin'], 
            actual_visual_right_edge + CONFIG['x_axis_margin'])

# 脊柱样式
for spine in ax.spines.values():
    spine.set_color('#BDBDBD')
    spine.set_linewidth(1.2)
    spine.set_alpha(0.9)
    
ax.tick_params(axis='both', which='major', labelsize=CONFIG['font_size_tick'], colors='#616161', width=1.2, length=5)

# ==========================================
# 7. 图例设置
# ==========================================
legend_elements = []
for i in range(len(metric_labels)):
    legend_elements.append(
        mpatches.Rectangle((0, 0), 1, 1, 
                           facecolor=CONFIG['colors'][i], 
                           edgecolor=CONFIG['edge_color'], 
                           linewidth=1.5,
                           hatch=CONFIG['hatch_patterns'][i],
                           label=metric_labels[i]))

ax.legend(handles=legend_elements,
          loc='upper center',
          bbox_to_anchor=(0.5, CONFIG['legend_y_pos']),
          fontsize=CONFIG['font_size_legend'],
          frameon=True,
          framealpha=0.95,
          edgecolor='#BDBDBD',
          facecolor='white',
          ncol=2, 
          columnspacing=1.5,
          handletextpad=0.5
          )

plt.subplots_adjust(top=0.8)
plt.tight_layout()

# ==========================================
# 8. 保存
# ==========================================
filename = 'error_analysis_comparison_dynamic.pdf'
plt.savefig(filename, format='pdf', dpi=300, bbox_inches='tight', facecolor='white')
print(f"图表已保存为 '{filename}'")
plt.show()

### python bar1.py 