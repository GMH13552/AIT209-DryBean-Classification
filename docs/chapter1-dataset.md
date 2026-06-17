# 第一章 · 数据集概述

## 从哪里来

**Dry Bean Dataset** 来自 UCI Machine Learning Repository（机器学习领域最经典的数据集仓库之一）。2020 年由 Koklu 和 Ozkan 发布。

数据是怎么来的？研究人员用高分辨率相机拍下 7 种干豆的图像，然后通过计算机视觉算法，从每颗豆子的轮廓中提取了 16 个数值特征。

> 📄 Koklu, M. and Ozkan, I.A. (2020). "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques." *Computers and Electronics in Agriculture*, 174, 105507.

## 要分什么

任务是将干豆分为 **7 个品种**：

```
BARBUNYA   927 条  ┤ ████████                 9.7%
BOMBAY     361 条  ┤ ███                      3.8%  ← 最少
CALI      1151 条  ┤ ██████████              12.1%
DERMASON  2503 条  ┤ ██████████████████████   26.3%  ← 最多
HOROZ     1340 条  ┤ ████████████            14.1%
SEKER     1408 条  ┤ ████████████            14.8%
SIRA      1837 条  ┤ ████████████████        19.3%
```

> ⚠️ BOMBAY 和 DERMASON 的比例接近 **1:7**，存在明显的类别不平衡。

## 用什么分

16 个特征全部是数值型，可分为三类：

**📐 几何特征（7 个）** — 描述豆粒的物理尺寸
```
Area, Perimeter, MajorAxisLength, MinorAxisLength,
ConvexArea, EquivDiameter, roundness
```

**📏 比例特征（2 个）** — 比值和归一化量
```
AspectRation, Extent
```

**🔷 形状特征（7 个）** — 描述轮廓形态
```
Eccentricity, Solidity, Compactness, ShapeFactor1~4
```

## 特征量纲差异

一个关键观察——这些特征的数值尺度差异巨大：

```
Area           8,204 ~ 207,528    ← 万级别
Perimeter        413 ~ 2,438     ← 千级别
Solidity         0.55 ~ 0.99     ← 0-1 之间
Eccentricity     0.22 ~ 0.99     ← 0-1 之间
```

这种量纲差异意味着：如果不做标准化，大数值特征（如 Area）会完全主导 KNN 和 SVM 的距离计算。一个 10 万级别的 Area 差异，会让 0.1 级别的 Solidity 差异变得微不足道——尽管它们可能同样重要。

标准化（第三章会详细处理）正是为了解决这个问题。

## 数据划分

教师提供了预划分的三个文件：

| 文件 | 样本数 | 用途 |
|------|:---:|------|
| `Dry_Bean_Dataset_Dirty_train.csv` | 9,527 | 训练模型 |
| `Dry_Bean_Dataset_Dirty_val.csv` | 1,347 | 验证调参 |
| `Dry_Bean_Dataset_Dirty_test.csv` | 2,737 | 最终测试 |

注意文件名中的 **Dirty**——这是教师故意留下的线索。数据经过了刻意污染，需要在建模前清洗。这恰是下一章的主题。
