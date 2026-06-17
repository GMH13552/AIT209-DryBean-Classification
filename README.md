# 🫘 Dry Bean 多分类 — 机器学习全流程项目

> **AIT209 机器学习与项目实践 · 期末作业**  
> 数据集：Dry Bean Dataset (UCI, 2020)  
> 任务：7 种干豆的多分类识别

---

## 📋 目录

1. [数据集概览](#1-数据集概览)
2. [数据分析 (EDA)](#2-数据分析-eda)
3. [数据预处理 & 特征工程](#3-数据预处理--特征工程)
4. [多算法实验](#4-多算法实验)
5. [系统架构 & 使用说明](#5-系统架构--使用说明)
6. [实验结果汇总](#6-实验结果汇总)
7. [课程总结](#7-课程总结)

---

## 1. 数据集概览

### 来源
- **UCI Machine Learning Repository** (Koklu & Ozkan, 2020)
- 由高分辨率相机拍摄的干豆图像，通过计算机视觉提取特征
- 原始数据：13,611 条样本，16 个数值特征，7 个类别

### 7 种豆类

| 编号 | 类别 | 样本数 | 占比 |
|:---:|------|--------|------|
| 0 | BARBUNYA | 1,322 | 9.7% |
| 1 | BOMBAY | 522 | 3.8% |
| 2 | CALI | 1,630 | 12.0% |
| 3 | DERMASON | 3,546 | 26.1% |
| 4 | HOROZ | 1,928 | 14.2% |
| 5 | SEKER | 2,027 | 14.9% |
| 6 | SIRA | 2,636 | 19.4% |

> ⚠️ BOMBAY 样本最少（仅 522 条），存在类别不平衡问题

### 16 个特征

| 类别 | 特征 | 说明 |
|------|------|------|
| 几何/形状 | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1-4 | 豆粒的尺寸和形状描述 |
| 颜色 | 由图像分析提取 | 色彩特征 |

> ✅ 所有特征均为数值型，**无缺失值**，无需类别编码

---

## 2. 数据分析 (EDA)

### 2.1 数据质量问题

在进行建模前，我们首先识别并处理以下问题：

- **类别不平衡**：BOMBAY 仅占 3.8%，DERMASON 占 26.1%，比例约 1:7
- **特征量纲差异**：Area 可达数万，而 Extent/Solidity 在 0-1 之间
- **异常值**：通过箱线图 (IQR) 检测可能存在极值

### 2.2 探索性分析要点

```
[关键发现]
├── 特征分布：多数特征呈右偏分布，需检查是否需要变换
├── 相关性分析：Area 与 ConvexArea、Perimeter 高度相关 (r > 0.95)
├── 类别可分性：通过 PCA/t-SNE 可视化，部分类别（DERMASON vs SIRA）边界模糊
└── 异常值：BOMBAY 类别在多个特征上存在离群点
```

### 2.3 可视化图表清单

- [ ] 类别分布柱状图
- [ ] 特征分布直方图 (16个)
- [ ] 相关性热力图
- [ ] PCA 降维散点图
- [ ] 每个特征按类别的箱线图
- [ ] t-SNE 可视化

---

## 3. 数据预处理 & 特征工程

### 3.1 数据清洗

| 步骤 | 方法 | 说明 |
|------|------|------|
| 缺失值处理 | 无需处理 | 原始数据无缺失值 |
| 异常值处理 | IQR / Z-score | 对比有无剔除的模型效果 |
| 类别编码 | LabelEncoder | 0-6 整数编码 |

### 3.2 特征工程

```
特征工程流程：
├── 标准化：StandardScaler (z-score)
│   └── 所有特征缩放到均值0、标准差1
├── 降维（可选实验）：PCA
│   └── 对比不同维度下的分类效果
├── 特征选择（可选实验）：
│   ├── 基于方差：移除低方差特征
│   ├── 基于相关性：移除高相关特征 (r > 0.95)
│   └── 基于重要性：RF/XGBoost 特征重要性排名
└── 数据增强（鲁棒性测试）：
    ├── 高斯噪声
    ├── 标签噪声（随机翻转）
    └── 特征缺失（随机遮蔽）
```

### 3.3 训练/测试集划分

由教师预先划分（具体比例待确认）。

---

## 4. 多算法实验

### 4.1 算法列表

| # | 算法 | 类型 | 课堂/课外 | 状态 |
|:---:|------|------|:---:|:---:|
| 1 | KNN | 基于距离 | 课堂 | ⬜ |
| 2 | SVM (RBF) | 核方法 | 课堂 | ⬜ |
| 3 | Random Forest | 集成学习 | 课堂 | ⬜ |
| 4 | XGBoost | 集成学习 | 课外✨ | ⬜ |
| 5 | MLP (神经网络) | 深度学习 | 课外✨ | ⬜ |
| 6 | LightGBM | 集成学习 | 课外✨ | ⬜ |

> 规则：至少 3 种算法，包含至少 1 种课外自学的

### 4.2 评估维度

#### (a) 测试集精度对比
- Accuracy, Precision, Recall, F1-score
- 混淆矩阵
- 各类别单独精度

#### (b) Loss 曲线对比
- 针对训练型算法（MLP, XGBoost）：绘制 train/val loss 曲线
- 非训练型（KNN）：无需

#### (c) 推理速度对比
- 统一硬件环境下，1000 次推理耗时统计

#### (d) 鲁棒性分析 🔥
训练数据加入不同噪声后，测试集精度下降对比：

| 噪声类型 | 强度 | KNN | SVM | RF | XGBoost | MLP |
|----------|------|:---:|:---:|:---:|:---:|:---:|
| 无噪声 (baseline) | 0 | - | - | - | - | - |
| 高斯噪声 | σ=0.1 | | | | | |
| 高斯噪声 | σ=0.5 | | | | | |
| 标签噪声 | 5% | | | | | |
| 标签噪声 | 10% | | | | | |
| 特征缺失 | 10% | | | | | |

#### (e) 过拟合分析
- 训练集 vs 测试集精度差异
- 差异越小 → 过拟合越轻

### 4.3 实验矩阵

```
                Accuracy  Precision  Recall  F1   Train-Test Gap  Speed(ms)
KNN               xx%       xx%      xx%    xx       xx%          xx
SVM               xx%       xx%      xx%    xx       xx%          xx
Random Forest     xx%       xx%      xx%    xx       xx%          xx
XGBoost           xx%       xx%      xx%    xx       xx%          xx
MLP               xx%       xx%      xx%    xx       xx%          xx
LightGBM          xx%       xx%      xx%    xx       xx%          xx
```

---

## 5. 系统架构 & 使用说明

### 5.1 项目结构

```
AIT209-DryBean-Classification/
├── data/                   # 数据集（不提交到 git）
│   ├── train.csv
│   └── test.csv
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载模块
│   ├── preprocess.py       # 预处理 & 特征工程
│   ├── train.py            # 训练入口
│   ├── evaluate.py         # 评估 & 对比
│   └── models/             # 各算法实现
│       ├── knn_model.py
│       ├── svm_model.py
│       ├── rf_model.py
│       ├── xgb_model.py
│       ├── mlp_model.py
│       └── lgbm_model.py
├── notebooks/              # Jupyter notebooks (EDA)
│   └── eda.ipynb
├── models/                 # 保存的模型权重
├── results/                # 实验结果（图表、表格）
│   ├── figures/
│   └── tables/
└── README.md
```

### 5.2 命令行使用

```bash
# 安装依赖
pip install -r requirements.txt

# 数据预处理
python src/preprocess.py --data_dir ./data

# 训练所有模型
python src/train.py --model all

# 训练指定模型
python src/train.py --model xgboost

# 评估 & 生成对比报告
python src/evaluate.py --results_dir ./results
```

### 5.3 依赖库

```
numpy, pandas, scikit-learn, matplotlib, seaborn
xgboost, lightgbm, torch (for MLP)
```

---

## 6. 实验结果汇总

### 6.1 最佳模型

| 排名 | 算法 | 测试精度 | F1-Score | 推理速度 |
|:---:|------|:---:|:---:|:---:|
| 🥇 | *待训练* | - | - | - |
| 🥈 | *待训练* | - | - | - |
| 🥉 | *待训练* | - | - | - |

### 6.2 关键发现

> *本部分在实验完成后填写*

1. 
2. 
3. 

### 6.3 图表展示

> *截图和可视化结果将在此展示*

---

## 7. 课程总结

### 7.1 学习收获

通过本课程和项目实践，掌握了：

- 完整的机器学习项目流程：从数据分析、特征工程、模型训练到评估部署
- 多种分类算法的原理和适用场景（KNN, SVM, RF, XGBoost, MLP）
- 模型评估的多个维度：不仅看精度，还要考虑鲁棒性、过拟合、推理速度
- 工程项目化能力：模块化代码、命令行接口、版本控制

### 7.2 课程评价与建议

> *待填写*

---

## 📎 附录

- **数据集来源**: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **论文引用**: Koklu, M. and Ozkan, I.A., (2020). "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques." *Computers and Electronics in Agriculture*, 174, 105507.
- **GitHub**: [本项目链接](https://github.com/GMH13552/AIT209-DryBean-Classification)

---

*最后更新：2026-06-17*
