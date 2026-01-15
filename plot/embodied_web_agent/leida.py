import matplotlib.pyplot as plt
import numpy as np

# ===========================
# 0. 全局配置
# ===========================
FONT_FAMILY = 'Times New Roman'
FONT_SIZE_LABEL = 25           # 指标名称
FONT_SIZE_LIMIT = 22           # 指标最大值标注
FONT_SIZE_VALUE = 22           # 数据点具体数值
FONT_SIZE_LEGEND = 24          # 图例
LINE_WIDTH = 3.0               # 线宽
GRID_ALPHA = 0.3               # 网格透明度

# --- 距离与位置控制 ---
LABEL_PAD = 30                 # 标签名称距离圆环边缘的距离
LIMIT_PAD = 103                # 最大值刻度距离圆心的绝对距离
DATA_LABEL_OFFSET = 9        # 数据点数字距离点的距离
ROTATION_CLOCKWISE_DEG = -22.5 # 雷达图顺时针旋转的角度

# --- 功能开关 ---
SHOW_LIMIT_VALUES = False      # 是否显示轴的最大刻度值

# 颜色定义
COLOR_REACT = '#2E86DE'
COLOR_WORLDMIND = '#EE5A6F'
COLOR_GRID = '#95A5A6'
COLOR_TEXT = '#2C3E50'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [FONT_FAMILY]

# ===========================
# 1. 数据与自定义范围
# ===========================
metrics = ['Overall\nAcc', 'Completion\nRate', 'Web Acc', 'Embodied\nAcc']

# 原始数据
raw_react = [9.82, 17.02, 33.93, 29.99, 11.61, 21.88, 45.54, 32.17]
raw_worldmind = [20.54, 39.99, 45.54, 48.70, 24.11, 41.50, 80.36, 47.27]

# 定义每个轴的最大刻度值 (Max Limits)
max_limits = [30, 55, 55, 65, 37, 65, 100, 65] 

# 【关键修改】动态计算每个轴的最小刻度值 (Min Limits)
# 规则：取两个数据中的最小值，然后减去 5
min_limits = [min(r, w) - 20 for r, w in zip(raw_react, raw_worldmind)]

labels = metrics + metrics

# ===========================
# 2. 数据归一化处理 (修改为区间映射)
# ===========================
def normalize(data, max_limits, min_limits):
    norm_list = []
    for d, max_l, min_l in zip(data, max_limits, min_limits):
        # 映射公式：(值 - 最小值) / (最大值 - 最小值) * 100
        # 结果为 0-100 之间的值，0代表圆心(min_l)，100代表边缘(max_l)
        rng = max_l - min_l
        if rng == 0:
            norm_list.append(0)
        else:
            norm_list.append((d - min_l) / rng * 100)
    return norm_list

values_react_norm = normalize(raw_react, max_limits, min_limits)
values_worldmind_norm = normalize(raw_worldmind, max_limits, min_limits)

# 闭合数据 (追加第一个点)
values_react_norm += values_react_norm[:1]
values_worldmind_norm += values_worldmind_norm[:1]
raw_react_closed = raw_react + raw_react[:1]
raw_worldmind_closed = raw_worldmind + raw_worldmind[:1]

N = len(raw_react)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ===========================
# 3. 绘图设置
# ===========================
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection='polar'))

# --- 旋转控制 ---
rotation_rad = np.radians(ROTATION_CLOCKWISE_DEG)
ax.set_theta_offset(-rotation_rad) 
ax.set_theta_direction(1)

# 去除默认边框和网格
ax.spines['polar'].set_visible(False)
ax.yaxis.grid(False)
ax.xaxis.grid(False) 

# ===========================
# 4. 手动绘制背景
# ===========================
grid_ratios = [25, 50, 75, 100] 

for r in grid_ratios:
    style = '-' if r == 100 else '--'
    width = 1.5 if r == 100 else 1.0
    alpha = 0.6 if r == 100 else GRID_ALPHA
    color = '#BDC3C7' if r == 100 else COLOR_GRID
    
    ax.plot(angles, [r]*len(angles), color=color, linestyle=style, 
            linewidth=width, alpha=alpha, zorder=0)

# 手动绘制放射线
for angle in angles[:-1]:
    ax.plot([angle, angle], [0, 100], color=COLOR_GRID, linestyle='--', alpha=GRID_ALPHA, lw=1)

# ===========================
# 5. 绘制数据
# ===========================
# ReAct
ax.plot(angles, values_react_norm, color=COLOR_REACT, linewidth=LINE_WIDTH, 
        marker='o', markersize=8, label='ReAct', markeredgewidth=1.5, 
        markeredgecolor='white', zorder=3)
ax.fill(angles, values_react_norm, color=COLOR_REACT, alpha=0.12)

# WorldMind
ax.plot(angles, values_worldmind_norm, color=COLOR_WORLDMIND, linewidth=LINE_WIDTH, 
        marker='D', markersize=7, label='WorldMind', markeredgewidth=1.5, 
        markeredgecolor='white', zorder=3)
ax.fill(angles, values_worldmind_norm, color=COLOR_WORLDMIND, alpha=0.12)

# ===========================
# 6. 数值标注
# ===========================
def add_labels(angles_seq, val_norm_seq, val_raw_seq, color, offset_dir=1):
    for angle, v_norm, v_raw in zip(angles_seq[:-1], val_norm_seq[:-1], val_raw_seq[:-1]):
        distance = v_norm + (DATA_LABEL_OFFSET * offset_dir)
        ax.text(angle, distance, f'{v_raw:.2f}', 
                color=color, size=FONT_SIZE_VALUE, 
                fontweight='bold', ha='center', va='center', zorder=5)

# 统一向外标注
add_labels(angles, values_react_norm, raw_react_closed, COLOR_REACT, offset_dir=1.2)
add_labels(angles, values_worldmind_norm, raw_worldmind_closed, COLOR_WORLDMIND, offset_dir=1.2)

# ===========================
# 7. 轴标签与最大值标注
# ===========================
ax.set_xticks(angles[:-1])
ax.set_xticklabels([]) 
ax.set_yticks([])

for i, (angle, label, max_limit) in enumerate(zip(angles[:-1], labels, max_limits)):
    
    # --- 位置计算 ---
    limit_pos = LIMIT_PAD         # 最大值刻度位置
    label_pos = 100 + LABEL_PAD   # 标签名位置
    
    # 绘制指标名
    ax.text(angle, label_pos, label, size=FONT_SIZE_LABEL, 
            fontweight='bold', color=COLOR_TEXT, ha='center', va='center')
    
    # 绘制最大值刻度
    if SHOW_LIMIT_VALUES:
        # 可以选择显示范围，例如: f"/{max_limit}" 或 f"{min_limits[i]:.0f}-{max_limit}"
        # 这里保持原样只显示最大值，如需显示最小值可改为: f"[{min_limits[i]:.0f}, {max_limit}]"
        ax.text(angle, limit_pos, f"/{max_limit}", size=FONT_SIZE_LIMIT, 
                color='#7F8C8D', ha='center', va='center', fontweight='normal')

# ===========================
# 8. 图例与保存
# ===========================
legend = plt.legend(loc='upper right', bbox_to_anchor=(1.55, 1.45), 
                   frameon=False, fontsize=FONT_SIZE_LEGEND)

plt.tight_layout()
plt.savefig('radar_comparison.png', format='png', dpi=300, bbox_inches='tight')
print("文件已成功保存为 'radar_comparison.pdf'")
plt.show()

### python leida.py