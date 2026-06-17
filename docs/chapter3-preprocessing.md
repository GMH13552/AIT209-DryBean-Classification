# 3. 数据预处理 & 特征工程

## 3.1 处理流水线

```
原始数据 (13611条 × 17列)
    │
    ├── ① 标签清洗 → 25 种变体 → 7 个标准类名
    ├── ② Solidity 修复 → "?" → NaN → 中位数填充
    ├── ③ Compactness 修复 → 去" cm" → float
    ├── ④ Perimeter 缺失值 → 中位数填充
    ├── ⑤ LabelEncoder → 0-6 整数编码
    ├── ⑥ StandardScaler → 均值 0，标准差 1
    │
    ▼
干净数据: X_train=(9527,16) / X_val=(1347,16) / X_test=(2737,16)
```

## 3.2 各步骤详解

### 步骤 ①：标签清洗

```python
class_map = {
    # 大小写修正
    'dermason': 'DERMASON', 'sira': 'SIRA',
    'horoz': 'HOROZ', 'seker': 'SEKER',
    'barbunya': 'BARBUNYA', 'cali': 'CALI', 'bombay': 'BOMBAY',
    # 数字替换字母修正
    'D3RMAS0N': 'DERMASON', 'H0R0Z': 'HOROZ',
    'S3K3R': 'SEKER', 'B0MBAY': 'BOMBAY',
}
df['Class'] = df['Class'].str.strip().map(
    lambda x: class_map.get(x, x.upper())
)
```

### 步骤 ②：Solidity 修复

```python
# "?" → NaN → 中位数填充
df['Solidity'] = pd.to_numeric(df['Solidity'], errors='coerce')
# coerce 会将非数值（"?"）转为 NaN
```

### 步骤 ③：Compactness 修复

```python
df['Compactness'] = (
    df['Compactness']
    .astype(str)
    .str.replace(' cm', '', regex=False)  # 去单位
    .str.strip()
)
df['Compactness'] = pd.to_numeric(df['Compactness'], errors='coerce')
```

### 步骤 ④~⑥：缺失值填充 + 编码 + 标准化

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 标签编码
le = LabelEncoder()
y = le.fit_transform(df['Class'])

# 提取特征
X = df.drop(columns=['Class'])

# 中位数填充
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 3.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 缺失值策略 | **中位数填充** | 对异常值鲁棒，优于均值；数据无极端偏斜 |
| 缩放方式 | **StandardScaler** | SVM/MLP/KNN 需要标准化；树模型不受影响 |
| 异常值处理 | **暂不剔除** | 保留信息完整性，避免过度清洗 |
| 降维 | **不做 PCA** | 16 个特征不算多，保留可解释性；可作对比实验 |
| 类别不平衡 | **暂不处理** | 先看模型自然处理能力 |

## 3.4 预处理后的数据统计

| 指标 | 清洗前 | 清洗后 |
|------|:---:|:---:|
| 特征列数 | 16 | 16 |
| 缺失值 | 671 个 | 0 |
| 标签类别数 | 25 种变体 | 7 个标准类名 |
| 数据类型 | 混合 (object+float+int) | 全部 float64 |
| 特征量纲 | 差异巨大 (10^5 range) | 均值 0，标准差 1 |

## 3.5 代码组织

预处理逻辑封装在 `src/preprocess.py` 中，对外暴露单一接口：

```python
from preprocess import preprocess_pipeline

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = \
    preprocess_pipeline()
```

这样可以保证训练、验证、测试三个阶段使用**完全相同的预处理参数**（同一个 imputer 和 scaler），避免数据泄露。
