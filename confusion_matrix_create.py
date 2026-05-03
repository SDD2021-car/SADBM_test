import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =========================
# 1. 指定 Times New Roman 字体路径
# =========================
font_path = '/data/yjy_data/fonts/times.ttf'

if not os.path.exists(font_path):
    raise FileNotFoundError(f'字体文件不存在: {font_path}')

# 把字体加入 matplotlib 字体管理器
fm.fontManager.addfont(font_path)

# 创建字体属性
times_new_roman = fm.FontProperties(fname=font_path)

print("当前加载字体:", times_new_roman.get_name())

# 全局设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [times_new_roman.get_name()]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


# =========================
# 2. 输入混淆矩阵数据
# =========================
labels = ['Farmland', 'Forest', 'Mountain', 'Rural', 'Semi-Urban', 'Urban']

cm = np.array([
    [143,0,0,17,0,0],
    [0,157,0,0,0,0],
    [0,0,151,0,0,0],
    [18,1,0,145,6,1],
    [0,0,0,10,124,15],
    [0,0,0,0,11,147]
])


# =========================
# 3. 绘制函数
# =========================
def plot_confusion_matrix(cm, labels, normalize=False, title='Confusion Matrix', save_path=None):
    if normalize:
        cm_to_plot = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    else:
        cm_to_plot = cm

    fig, ax = plt.subplots(figsize=(6, 4))

    # 蓝白配色
    im = ax.imshow(cm_to_plot, interpolation='nearest', cmap=plt.cm.Blues)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)

    if normalize:
        cbar.set_label(
            'Proportion',
            fontsize=14,
            fontproperties=times_new_roman
        )
    else:
        cbar.set_label(
            'Number of samples',
            fontsize=14,
            fontproperties=times_new_roman
        )

    # 强制颜色条刻度字体
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontproperties(times_new_roman)
        tick.set_fontsize(14)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # 强制坐标轴标题字体
    ax.set_xlabel(
        'Predicted label',
        fontsize=12,
        fontproperties=times_new_roman
    )
    ax.set_ylabel(
        'True label',
        fontsize=12,
        fontproperties=times_new_roman
    )
    ax.set_title(
        title,
        fontsize=16,
        fontproperties=times_new_roman
    )

    # 强制 x 轴刻度字体
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(times_new_roman)
        tick.set_fontsize(14)
        tick.set_rotation(45)
        tick.set_ha('right')
        tick.set_rotation_mode('anchor')

    # 强制 y 轴刻度字体
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(times_new_roman)
        tick.set_fontsize(14)

    # 强制坐标轴数字刻度字体
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontproperties(times_new_roman)

    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontproperties(times_new_roman)

    # 在每个单元格中标注数值
    threshold = cm_to_plot.max() / 2.0

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            if normalize:
                value = cm_to_plot[i, j]
                text = f'{value:.2f}'
            else:
                value = int(cm_to_plot[i, j])
                text = f'{value}'

            ax.text(
                j, i, text,
                ha='center',
                va='center',
                color='white' if value > threshold else 'black',
                fontsize=14,
                fontproperties=times_new_roman
            )

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# =========================
# 4. 绘制原始混淆矩阵
# =========================
plot_confusion_matrix(
    cm=cm,
    labels=labels,
    normalize=False,
    title='Confusion Matrix of only SAR Results\nBased on the ResNet-18 Classification Network',
    save_path='/data/yjy_data/DDBM_GT_Unet/confusion_matrix/confusion_matrix_blue_white_res18_SAR.png'
)

# =========================
# 5. 绘制归一化混淆矩阵
# =========================
plot_confusion_matrix(
    cm=cm,
    labels=labels,
    normalize=True,
    title='Normalized Confusion Matrix of only SAR Results\nBased on the ResNet-18 Classification Network',
    save_path='/data/yjy_data/DDBM_GT_Unet/confusion_matrix/normalized_confusion_matrix_blue_white_res18_SAR.png'
)