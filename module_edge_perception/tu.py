import matplotlib.pyplot as plt
import numpy as np

# --- 1. 数据准备 ---
# 根据图片估算的数据点
# X轴的类别
labels = ['5', '10', '15', '20', '25', '30']
# 四个系列的数据
mrs_data = [0.43, 0.35, 0.29, 0.27, 0.21, 0.20]
adainf_data = [0.81, 0.82, 0.80, 0.87, 0.84, 0.81]
ekya_data = [0.86, 0.88, 0.90, 0.91, 0.88, 0.86]
pass_data = [0.47, 0.37, 0.31, 0.30, 0.24, 0.22]


# --- 2. 绘图设置 ---
# 设置字体以更好地匹配原图
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] # 优先使用Arial字体

# X轴的位置
x = np.arange(len(labels))
# 每个柱子的宽度
width = 0.2

# 创建画布和坐标轴
fig, ax = plt.subplots(figsize=(10, 7)) # 设置图像大小

# --- 3. 绘制柱状图 ---
rects1 = ax.bar(x - 1.5*width, mrs_data, width, label='ORRIC',
                color='#263b5e', hatch='/',edgecolor='white')


rects2 = ax.bar(x - 0.5*width, adainf_data, width, label='zyf',
                color='#0073bd', hatch='+', edgecolor='white')


rects3 = ax.bar(x + 0.5*width, ekya_data, width, label='Ekya',
                color='#86a9c1', hatch='x', edgecolor='white')


rects4 = ax.bar(x + 1.5*width, pass_data, width, label='PASS',
                color='#a6d9f5', hatch='\\',edgecolor='white')

# --- 4. 美化图表 ---
# 设置坐标轴标签和标题
ax.set_ylabel('Normalized Completion Time', fontsize=20)
ax.set_xlabel('Number of Applications', fontsize=20)

# 设置坐标轴刻度
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_yticks(np.arange(0, 1.0, 0.2))
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 1.0) # 设置Y轴范围

# 添加图例
ax.legend(fontsize=16)

# 添加网格线
ax.grid(True, zorder=0) # zorder=0 将网格线置于底层

# 调整布局以防止标签重叠
fig.tight_layout()

# --- 5. 显示图像 ---
plt.show()