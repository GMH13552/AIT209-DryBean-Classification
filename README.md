# 🫘 Dry Bean 多分类 — 机器学习全流程项目

> **AIT209 机器学习与项目实践 · 期末作业**  
> 数据集：Dry Bean Dataset (UCI, 2020)  
> 任务：7 种干豆的多分类识别  
> **GitHub**: https://github.com/GMH13552/AIT209-DryBean-Classification

---

## 📋 目录

1. [数据集概览](#1-数据集概览)
2. [数据分析 & 数据质量问题](#2-数据分析--数据质量问题)
3. [数据预处理 & 特征工程](#3-数据预处理--特征工程)
4. [算法实现 & 实验结果](#4-算法实现--实验结果)
5. [系统架构 & 使用说明](#5-系统架构--使用说明)
6. [课程总结](#6-课程总结)

---

## 1. 数据集概览

### 来源
- **UCI Machine Learning Repository** (Koklu & Ozkan, 2020)
- 由高分辨率相机拍摄的干豆图像，通过计算机视觉提取特征
- 教师预先分为训练集 (9527)、验证集 (1347)、测试集 (2737)

### 7 种豆类

| 编号 | 类别 | 样本数 | 占比 |
|:---:|------|--------|------|
| 0 | BARBUNYA | 927 | 9.7% |
| 1 | BOMBAY | 361 | 3.8% |
| 2 | CALI | 1,151 | 12.1% |
| 3 | DERMASON | 2,503 | 26.3% |
| 4 | HOROZ | 1,340 | 14.1% |
| 5 | SEKER | 1,408 | 14.8% |
| 6 | SIRA | 1,837 | 19.3% |

> ⚠️ BOMBAY 样本最少（仅 361 条），存在类别不平衡问题

### 16 个特征

| 类别 | 特征 | 说明 |
|------|------|------|
| 几何/形状 (12个) | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1-4 | 豆粒的尺寸和形状描述 |
| 纵横比 (1个) | AspectRation | 长轴/短轴比 |
| 偏心率 (1个) | Eccentricity | 形状偏离圆形的程度 |
| 无法分类 (2个) | ShapeFactor3, ShapeFactor4 | 额外形状因子 |

---

## 2. 数据分析 & 数据质量问题

### 2.1 发现的数据污染 （教师刻意注入）

| 问题 | 字段 | 具体情况 | 处理方式 |
|------|------|----------|----------|
| **❌ 非法字符串** | Solidity | 包含 `?` 值（train 202个） | 替换为 NaN → 中位数填充 |
| **❌ 单位混入** | Compactness | 部分值附带 ` cm` 后缀（如 `0.8194 cm`） | 剥离 ` cm` → 转 float |
| **❌ 缺失值** | Perimeter | train 469个 / val 62个 / test 146个 NaN | 中位数填充 |
| **❌ 标签混乱** | Class | 大小写 + 拼写错误共 **25 种变体** | 标准化映射 |

### 2.2 标签混乱详情

原始数据中 7 个类别出现了 25 种写法：

| 正确标签 | 发现的变体 |
|----------|-----------|
| DERMASON | `dermason`, `D3RMAS0N`, `DERMASON `（空格） |
| SIRA | `sira` |
| HOROZ | `horoz`, `H0R0Z` |
| SEKER | `seker`, `S3K3R` |
| BARBUNYA | `barbunya` |
| CALI | `cali` |
| BOMBAY | `bombay`, `B0MBAY` |

### 2.3 数据清洗代码示例

```python
# 1. 修复 Solidity — "?" → NaN
df['Solidity'] = pd.to_numeric(df['Solidity'], errors='coerce')

# 2. 修复 Compactness — 去 " cm" 后缀
df['Compactness'] = df['Compactness'].str.replace(' cm', '').astype(float)

# 3. 标准化标签 — 映射到标准类名
class_map = {'dermason': 'DERMASON', 'D3RMAS0N': 'DERMASON',
             'horoz': 'HOROZ', 'H0R0Z': 'HOROZ', ...}
df['Class'] = df['Class'].map(lambda x: class_map.get(x, x.upper()))

# 4. 中位数填充缺失值
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
```

---

## 3. 数据预处理 & 特征工程

### 3.1 处理流程

```
原始数据 (13611条 × 17列)
    │
    ├── ① 标签清洗 → 25种变体 → 7个标准类名
    ├── ② Solidity 修复 → "?" → NaN → 中位数填充
    ├── ③ Compactness 修复 → 去" cm" → float
    ├── ④ Perimeter 缺失值 → 中位数填充
    ├── ⑤ LabelEncoder → 0-6 整数编码
    ├── ⑥ StandardScaler → 均值0，标准差1
    │
    ▼
干净数据: X_train=(9527,16) / X_val=(1347,16) / X_test=(2737,16)
```

### 3.2 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 缺失值策略 | 中位数 | 对异常值鲁棒，优于均值 |
| 缩放方式 | StandardScaler | SVM/MLP 需要标准化，树模型不受影响 |
| 异常值 | 暂不剔除 | 保留信息完整性 |
| 降维 | 暂不做 PCA | 16 个特征不算多，保留可解释性 |

---

## 4. 算法实现 & 实验结果

### 4.1 实现的算法（6种）

| # | 算法 | 类型 | 课堂/课外 |
|:---:|------|------|:---:|
| 1 | KNN (k=5) | 基于距离 | 课堂 |
| 2 | SVM (RBF, C=10) | 核方法 | 课堂 |
| 3 | Random Forest (100棵树, depth=12) | 集成学习 | 课堂 |
| 4 | Gradient Boosting (100轮, depth=6) | 集成提升 | 课外✨ |
| 5 | HistGradient Boosting (100轮, depth=8) | 集成提升 | 课外✨ |
| 6 | MLP (50→25, ReLU) | 神经网络 | 课外✨ |

> 标记 ✨ 的是课堂上没讲过的算法（3种课外自学）

### 4.2 测试集精度对比

| 排名 | 算法 | 测试精度 | F1-Score | 训练时间 |
|:---:|------|:---:|:---:|:---:|
| 🥇 | **SVM (RBF)** | **93.31%** | 0.9332 | 0.6s |
| 🥈 | Gradient Boosting | 92.62% | 0.9263 | 60.2s |
| 🥉 | HistGrad Boosting | 92.47% | 0.9248 | 5.8s |
| 4 | MLP | 92.40% | 0.9242 | 1.2s |
| 5 | Random Forest | 92.36% | 0.9237 | 1.3s |
| 6 | KNN | 91.78% | 0.9179 | 0.001s |

### 4.3 过拟合分析

| 算法 | 训练精度 | 验证精度 | 测试精度 | 过拟合 Gap |
|------|:---:|:---:|:---:|:---:|
| SVM (RBF) | 93.67% | 92.72% | 93.31% | **+0.95%** ✅ |
| MLP | 93.01% | 92.80% | 92.40% | **+0.21%** ✅✅ |
| KNN | 93.86% | 91.61% | 91.78% | +2.25% |
| Random Forest | 98.32% | 91.91% | 92.36% | +6.41% ⚠️ |
| Gradient Boosting | 100.00% | 92.43% | 92.62% | +7.57% ❌ |
| HistGrad Boosting | 100.00% | 91.91% | 92.47% | +8.09% ❌ |

> **分析**：SVM 和 MLP 泛化最好，过拟合最低。Boosting 类算法在训练集上完全拟合（100%），存在明显过拟合。

### 4.4 推理速度对比

| 算法 | 单次推理耗时 (ms) | 相对速度 |
|------|:---:|:---:|
| MLP | 1.9ms | 🚀 最快 |
| Random Forest | 31.7ms | ⚡ 快 |
| Gradient Boosting | 63.7ms | ⚡ 快 |
| HistGrad Boosting | 183.1ms | 🐢 较慢 |
| SVM (RBF) | 453.1ms | 🐌 慢 |
| KNN | 480.6ms | 🐌 最慢 |

> KNN 和 SVM 推理慢是因为需要存储全部训练数据 / 支持向量。

---

## 5. 系统架构 & 使用说明

### 5.1 项目结构

```
AIT209-DryBean-Classification/
├── data/                   # 数据集（已 gitignore，不提交）
│   ├── Dry_Bean_Dataset_Dirty_train.csv
│   ├── Dry_Bean_Dataset_Dirty_val.csv
│   └── Dry_Bean_Dataset_Dirty_test.csv
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载 & 标签映射
│   ├── preprocess.py       # 预处理 & 特征工程流水线
│   ├── run.py              # 训练 & 评估入口
│   └── train.py            # 完整训练流水线（含鲁棒性测试）
├── results/                # 实验结果
│   └── model_comparison.csv
├── models/                 # 模型保存目录
├── requirements.txt        # 依赖清单
└── README.md               # 项目文档
```

### 5.2 命令行使用

```bash
# 安装依赖
pip install -r requirements.txt

# 数据预处理（单独运行）
py src/preprocess.py

# 训练 & 评估所有模型
py src/run.py

# 完整流水线（含鲁棒性分析）
py src/train.py
```

### 5.3 依赖库

```
numpy, pandas, scikit-learn, matplotlib, seaborn
```

---

## 6. 课程总结

### 6.1 学习收获

通过本课程和项目实践，我学到了：

1. **完整 ML 项目流程**：从数据清洗 → 特征工程 → 模型训练 → 多维度评估的端到端实践
2. **数据质量意识**：真实数据往往包含各种"污染"（缺失值、格式错误、标签混乱），清洗是建模前最关键的一步
3. **多种分类算法**：掌握了 KNN、SVM、随机森林、Gradient Boosting、MLP 的原理和 sklearn 实现
4. **No Free Lunch**：没有万能算法 — SVM 精度最高但推理慢，MLP 过拟合最低但需要调参，Boosting 在训练集上轻松 100% 但泛化差
5. **多维度评估**：不只盯着精度，还要看过拟合程度、推理速度、鲁棒性

### 6.2 课程评价与建议

- 课程实践性强，从数据清洗到算法对比，覆盖了 ML 工程师的日常工作
- "脏数据"设计很好，模拟了真实场景中的数据处理问题
- 建议：可以增加深度学习（PyTorch/TensorFlow）的入门实验

---

## 📎 附录

- **数据集来源**: [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **论文引用**: Koklu, M. and Ozkan, I.A., (2020). "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques." *Computers and Electronics in Agriculture*, 174, 105507.
- **GitHub**: https://github.com/GMH13552/AIT209-DryBean-Classification

---

*最后更新：2026-06-17 · 实验完成*
