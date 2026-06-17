# 第三章 · 清源：数据预处理

上一章我们发现了 4 类数据问题。这一章逐一解决，最终产出干净、标准化的数据。

整个预处理流水线被封装在 `src/preprocess.py` 的 `preprocess_pipeline()` 函数中。但我们不只看结果，更要理解每一步的**为什么**。

本项目的预处理流程共 5 步：

```
原始数据 → ①标签归一化 → ②Solidity修复 → ③Compactness修复 → ④缺失值填充 → ⑤标准化 → 干净数据
```

---

## 3.1 标签归一化

### 策略

建立一个映射表，把所有 25 种变体一一映射到 7 个标准类名。

### 实现

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

### 关键设计

- `str.strip()` 先去掉首尾空格，解决末尾空格问题
- 先查映射表，找不到的用 `x.upper()` 兜底（大写化）
- 这样可以处理未来可能出现的新变体

### 处理效果

```
处理前: 25 种标签变体（含大小写、数字替换、空格）
处理后: 7 个标准类名，全部大写
```

---

## 3.2 Solidity 修复

### 策略

`"?"` → `NaN` → 中位数填充。

### 实现

```python
# Step 1: "?" → NaN
df['Solidity'] = pd.to_numeric(df['Solidity'], errors='coerce')
# errors='coerce' 会把无法转换的值（包括 "?"）转为 NaN

# Step 2: NaN → 中位数（在 §3.4 统一处理）
```

### 为什么用 `pd.to_numeric(errors='coerce')`？

这是处理字符串污染数值列最简洁的方法。一行代码把能转的转成浮点数，不能转的自动变 `NaN`。不需要手写正则，不需要 `replace`。

---

## 3.3 Compactness 修复

### 策略

剥离 ` cm` 后缀 → 转 float。

### 实现

```python
df['Compactness'] = (
    df['Compactness']
    .astype(str)                        # 确保全是字符串
    .str.replace(' cm', '', regex=False) # 去单位后缀
    .str.strip()                        # 去多余空格
)
df['Compactness'] = pd.to_numeric(df['Compactness'], errors='coerce')
```

### 为什么不用正则？

`' cm'` 是一个固定字符串，用 `str.replace()` 比正则更快更安全。用 `regex=False` 明确告诉 pandas 这是字面量匹配。

---

## 3.4 缺失值填充

### 策略

使用 **中位数（median）** 填充所有数值列的缺失值。

### 实现

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_val   = imputer.transform(X_val)
X_test  = imputer.transform(X_test)
```

### 为什么是中位数而非均值？

| 策略 | 优点 | 缺点 |
|------|------|------|
| 均值 | 无偏估计 | 受异常值影响大 |
| **中位数** | **对异常值鲁棒** | 对正态分布偏差略大 |
| 众数 | 简单 | 连续变量不适用 |

对于现实数据（往往有异常值），中位数比均值更安全。

> ⚠️ **重要**：在训练集上 `fit`，在验证集和测试集上只用 `transform`——防止数据泄露。

---

## 3.5 标准化

### 策略

使用 `StandardScaler`，将每个特征缩放到均值 0、标准差 1。

### 为什么需要标准化？

Dry Bean 数据集的特征量纲差异巨大：

```
Area:         8,000 ~ 200,000   ← 大数值
Perimeter:      400 ~ 2,500    ← 中数值
Solidity:         0 ~ 1        ← 小数值
```

对于基于距离的算法（KNN、SVM），未标准化的数据会让大数值特征主导距离计算。

```
未标准化时 KNN 的距离 ≈ √(Area差² + Perimeter差² + ...)
                        ≈ √(100000² + ...)
                        ≈ 几乎完全由 Area 决定
    
标准化后                ≈ √(1.5² + 0.8² + ...)
                        ≈ 各特征平等贡献
```

对于树模型（Random Forest、AdaBoost），标准化不影响结果——但统一处理让代码简洁。

### 实现

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)
```

---

## 处理效果一览

```
处理前                          处理后
─────────────────────────────────────────────
25 种标签变体        →        7 个标准类名
Solidity: object     →        float64, 无 NaN
Compactness: object  →        float64, 无 "cm"
Perimeter: 469 NaN   →        全部填充完成
量纲差异巨大          →        均值 0, 标准差 1
```

---

## 代码组织

所有预处理逻辑封装在 `preprocess_pipeline()` 中，对外暴露清晰接口：

```python
from preprocess import preprocess_pipeline

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler, features = \
    preprocess_pipeline()
```

这个设计有三层含义：
1. **模块化**：主流程不关心预处理细节
2. **可复现**：每次调用走完全相同的路径
3. **防泄露**：同一个 imputer 和 scaler 用于全量数据
