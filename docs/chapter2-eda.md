# 2. 数据分析 & 数据污染

## 2.1 数据污染总览

教师提供的并非干净的 UCI 原始数据，而是**刻意注入了 4 类数据质量问题**的"脏数据"版本。这是本课程实践性的核心设计。

| # | 问题类型 | 受影响字段 | 严重程度 | 数量 (训练集) |
|:---:|----------|------------|:---:|:---:|
| 1 | 非法占位符 | Solidity | 🔴 高 | 202 个 `?` |
| 2 | 单位混入 | Compactness | 🟡 中 | 多处 ` cm` 后缀 |
| 3 | 缺失值 | Perimeter | 🔴 高 | 469 个 NaN |
| 4 | 标签混乱 | Class | 🟡 中 | 25 种变体 |

## 2.2 问题详情

### 问题 1：Solidity 含 `?` 值

```python
>>> df['Solidity'].dtype
dtype('O')  # object 类型，非数值！

>>> df['Solidity'].value_counts()['?']
202  # 训练集中 202 个问号
```

**原因**：`?` 是常见的数据缺失占位符（常用于 UCI 原始格式）。

### 问题 2：Compactness 含单位后缀

```python
>>> df['Compactness'].value_counts().head(5)
0.8194 cm             3
0.8058 cm             3
0.6921 cm             3
0.6785015334152743    2    # 部分正常值
0.6877689206199188    2
```

**原因**：部分值被错误地附带了 ` cm` 单位后缀，导致整列被识别为字符串。

### 问题 3：Perimeter 缺失值

| 数据集 | 缺失数量 | 占比 |
|--------|:---:|------|
| 训练集 | 469 | 4.9% |
| 验证集 | 62 | 4.6% |
| 测试集 | 146 | 5.3% |

**原因**：模拟真实数据采集中常见的传感器故障或人工录入遗漏。

### 问题 4：Class 标签混乱

原始数据中 7 个类别出现了 **25 种变体**：

| 正确标签 | 发现的全部变体 |
|----------|---------------|
| DERMASON | `dermason` `D3RMAS0N` `DERMASON `（末尾空格） |
| SIRA | `sira` `SIRA ` |
| HOROZ | `horoz` `H0R0Z` `HOROZ ` |
| SEKER | `seker` `S3K3R` `SEKER ` |
| BARBUNYA | `barbunya` `BARBUNYA ` |
| CALI | `cali` `CALI ` |
| BOMBAY | `bombay` `B0MBAY` `BOMBAY ` |

**错误模式**：
- 🔡 大小写混乱（`dermason` → `DERMASON`）
- 🔢 数字替换字母（`D3RMAS0N` → `DERMASON`，`H0R0Z` → `HOROZ`，`S3K3R` → `SEKER`，`B0MBAY` → `BOMBAY`）
- 📏 末尾多余空格（`DERMASON `）

## 2.3 数据质量诊断代码

```python
import pandas as pd

df = pd.read_csv('Dry_Bean_Dataset_Dirty_train.csv')

# 1. 检查数据类型
print(df.dtypes)
# → Solidity: object ❌  应该是 float64
# → Compactness: object ❌  应该是 float64

# 2. 检查缺失值
print(df.isnull().sum()[df.isnull().sum() > 0])
# → Perimeter: 469 个缺失

# 3. 检查标签分布
print(df['Class'].value_counts())
# → 25 个唯一值（应该有 7 个）

# 4. 检查非数值内容
non_numeric = df['Solidity'].apply(
    lambda x: x if not str(x).replace('.','').isdigit() else None
).dropna()
print(non_numeric.value_counts())
# → ?: 202
```

## 2.4 探索性分析要点

```
数据质量检查清单:
├── ✅ 无重复行
├── ❌ Solidity 含 202 个 ?
├── ❌ Compactness 含 cm 后缀
├── ❌ Perimeter 含 469 个缺失值
├── ❌ Class 标签含 25 种变体（应为 7 种）
├── ⚠️ 特征量纲差异巨大 (Area ~10^5, Extent ~0.5)
├── ⚠️ BOMBAY:DERNASON = 1:7 类别不平衡
└── ⚠️ Area 与 ConvexArea 预期高度相关
```
