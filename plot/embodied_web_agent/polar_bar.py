import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
CONFIG = {
    # 尺寸控制
    'inner_radius': 80,
    'text_margin': 11,             # 柱子上数值距离柱顶的距离
    'group_label_offset': 18,     # 内圈标签距离柱子底部的距离
    
    # 字体大小
    'font_size_title': 14,
    'font_size_group': 11,
    'font_size_value': 12,
    'font_size_legend': 13,
    
    # 颜色
    'colors': ['#A6C9EC', '#EAC8C3', '#F2E6E1', '#D8E6D6']
}

# ==========================================
# 2. 数据准备
# ==========================================
group_labels = ['GPT-3.5-turbo\nReAct', 'GPT-3.5-turbo\nWorldMind', 'GPT-4.1-mini\nReAct', 'GPT-4.1-mini\nWorldMind']
metric_labels = ['Max Steps Reached', 'Step Execution Error', 'Repeated Web Actions', 'Repeated Embodied Actions']

data = np.array([
    [68, 22, 17, 5],
    [93, 12, 0,  7],
    [73, 27, 12, 0],
    [96, 8,  0,  8]
])

# ==========================================
# 3. 绘图计算逻辑
# ==========================================
n_groups = len(group_labels) 
n_metrics = len(metric_labels)

# 基础角度划分 (每个组的中心位置)
group_centers = np.linspace(0, 2 * np.pi, n_groups, endpoint=False)
group_width = 2 * np.pi / n_groups

# 柱子宽度 (保持不变，确保所有柱子粗细一致)
bar_width = group_width / (n_metrics + 1.5)

inner_radius = CONFIG['inner_radius']
colors = CONFIG['colors']

# ==========================================
# 4. 开始绘图
# ==========================================
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1) 

# 用于收集图例句柄
legend_handles = {}

# --- 【核心修改】按组循环，实现动态居中 ---
for group_idx in range(n_groups):
    # 1. 找出当前组中非0的指标索引
    # data[group_idx] 是这一行的四个数据，例如 [93, 12, 0, 7]
    # np.where 返回的是一个 tuple，取 [0] 获取索引数组
    active_indices = np.where(data[group_idx] > 0)[0]
    
    # 2. 计算当前有多少个柱子要画
    n_active = len(active_indices)
    
    if n_active == 0: continue # 如果全是0，跳过该组

    # 3. 动态计算起始角度 (为了让这 n_active 个柱子居中)
    # 逻辑：组中心 - (总宽度的一半) + (半个柱宽)
    # 总宽度 = n_active * bar_width
    current_start_angle = group_centers[group_idx] - (n_active * bar_width) / 2 + (bar_width / 2)
    
    # 4. 循环绘制当前组的有效柱子
    for i, metric_idx in enumerate(active_indices):
        value = data[group_idx, metric_idx]
        
        # 计算当前柱子的具体角度
        angle = current_start_angle + i * bar_width
        
        # 获取对应的颜色和标签
        color = colors[metric_idx]
        label = metric_labels[metric_idx]
        
        # 绘制柱子
        bar = ax.bar(angle, value, width=bar_width, bottom=inner_radius, 
                     color=color, alpha=0.9, edgecolor='white', linewidth=0.5,
                     label=label)
        
        # 记录图例 (用于后面手动排序)
        if label not in legend_handles:
            legend_handles[label] = bar
            
        # 绘制数值
        text_pos = inner_radius + value + CONFIG['text_margin']
        ax.text(angle, text_pos, str(value), 
                ha='center', va='bottom', 
                fontsize=CONFIG['font_size_value'], 
                fontweight='bold', color='#333333')

# --- 内圈标签 ---
label_radius = inner_radius - CONFIG['group_label_offset']

for angle, label in zip(group_centers, group_labels):
    angle_deg = np.degrees(angle)
    rotation = angle_deg if not (90 < angle_deg < 270) else angle_deg + 180
        
    ax.text(angle, label_radius, label, ha='center', va='center', 
            fontsize=CONFIG['font_size_group'], 
            fontweight='bold', color='#333333', 
            rotation=rotation, rotation_mode='anchor')

# ==========================================
# 5. 去除坐标轴和空白
# ==========================================
ax.set_axis_off() 

# 动态计算Y轴上限
max_val = np.max(data)
ax.set_ylim(0, inner_radius + max_val + 10)

# 中心标题
ax.text(0, 0, 'Model\nComparison', ha='center', va='center', 
        fontsize=CONFIG['font_size_title'], 
        fontweight='bold', color=colors[0])

# --- 图例处理 (确保顺序正确) ---
# 因为我们是动态绘制的，直接用 plt.legend() 可能会乱序或者丢失
# 我们根据原始 metric_labels 的顺序重建 handle 列表
sorted_handles = [legend_handles[lbl] for lbl in metric_labels if lbl in legend_handles]
sorted_labels = [lbl for lbl in metric_labels if lbl in legend_handles]

plt.legend(
    sorted_handles, sorted_labels, 
    loc='upper center',            
    bbox_to_anchor=(0.5, 0.05),   
    ncol=2, 
    columnspacing=1.5,
    handletextpad=0.8,
    frameon=False, 
    fontsize=CONFIG['font_size_legend']
)

# ==========================================
# 6. 保存
# ==========================================
save_name = 'model_comparison_polar.pdf'
plt.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0.0)
print(f"PDF已保存为: {save_name} (已去除边缘空白，且动态移除空柱子)")

plt.show()


### python polar_bar.py