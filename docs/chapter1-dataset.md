# 1. 数据集概览

## 来源

**Dry Bean Dataset** 来自 UCI Machine Learning Repository，由 Koklu & Ozkan 于 2020 年发布。

数据采集过程：使用高分辨率相机拍摄 7 种干豆的图像，通过计算机视觉技术从每颗豆子的轮廓中提取 16 个数值特征。

- **原始论文**: Koklu, M. and Ozkan, I.A., "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques." *Computers and Electronics in Agriculture*, 174, 105507 (2020)
- **UCI 页面**: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

## 数据划分（教师提供）

| 数据集 | 样本数 | 占比 |
|--------|--------|------|
| 训练集 (train) | 9,527 | 70% |
| 验证集 (val) | 1,347 | 10% |
| 测试集 (test) | 2,737 | 20% |
| **合计** | **13,611** | 100% |

> 注：数据已由教师划分为三个文件，且**刻意注入了数据质量问题**（详见第 2 章）

## 7 种豆类

| 编号 | 类别名 | 训练集样本数 | 占比 |
|:---:|--------|:---:|------|
| 0 | BARBUNYA（红花豆） | 927 | 9.7% |
| 1 | BOMBAY（孟买豆） | 361 | 3.8% |
| 2 | CALI（卡利豆） | 1,151 | 12.1% |
| 3 | DERMASON（德马森豆） | 2,503 | 26.3% |
| 4 | HOROZ（霍罗兹豆） | 1,340 | 14.1% |
| 5 | SEKER（塞克尔豆） | 1,408 | 14.8% |
| 6 | SIRA（西拉豆） | 1,837 | 19.3% |

> ⚠️ **类别不平衡**：BOMBAY 仅 361 条 vs DERMASON 2,503 条，比例约 **1:7**

## 16 个特征

所有特征均为从豆粒图像中提取的数值型特征，无类别型变量：

| 特征 | 类型 | 说明 |
|------|------|------|
| Area | 几何 | 豆粒面积（像素²） |
| Perimeter | 几何 | 周长（像素） |
| MajorAxisLength | 几何 | 长轴长度 |
| MinorAxisLength | 几何 | 短轴长度 |
| AspectRation | 比例 | 长宽比 (Major/Minor) |
| Eccentricity | 形状 | 偏心率（0=圆，1=线段） |
| ConvexArea | 几何 | 凸包面积 |
| EquivDiameter | 几何 | 等效直径 |
| Extent | 比例 | 面积/边界框面积 |
| Solidity | 比例 | 面积/凸包面积 |
| roundness | 形状 | 圆度 |
| Compactness | 形状 | 紧密度 |
| ShapeFactor1 | 形状 | 形状因子 1 |
| ShapeFactor2 | 形状 | 形状因子 2 |
| ShapeFactor3 | 形状 | 形状因子 3 |
| ShapeFactor4 | 形状 | 形状因子 4 |

### 特征量纲差异

| 特征 | 典型范围 | 类型 |
|------|----------|------|
| Area, ConvexArea | 8,000 ~ 200,000 | 大数值 |
| Perimeter | 400 ~ 2,500 | 中数值 |
| MajorAxisLength, MinorAxisLength | 100 ~ 800 | 中数值 |
| EquivDiameter | 100 ~ 500 | 中数值 |
| Extent, Solidity, roundness | 0 ~ 1 | 小数 |
| Compactness, ShapeFactor1-4 | 0 ~ 1 | 小数 |
| AspectRation | 1 ~ 3 | 小数 |
| Eccentricity | 0 ~ 1 | 小数 |

> 💡 量纲差异巨大 → 必须标准化才能正确使用 SVM、MLP 等距离敏感算法
