# 第二章 · 辨污：数据分析

拿到数据后，第一件事不是急着建模，而是**看看手里到底是什么**。数据分析的目标是发现数据中的问题，理解数据的特征，为后续处理做准备。

本章采用**逐步诊断**的方式，一个问题一个问题地排查，像侦探破案一样揭开数据的面纱。

---

## 2.1 第一眼：数据类型诊断

用 `pd.read_csv()` 加载训练集后，第一个动作是检查各列的数据类型：

```python
>>> df.dtypes
Area                int64      ← 数值，OK
Perimeter         float64      ← 数值，OK
...
Solidity             object    ← ❌ 应该是 float64！
Compactness          object    ← ❌ 应该是 float64！
Class                object    ← ✅ 类别标签，object 正常
```

立刻发现两个异常：`Solidity` 和 `Compactness` 本应是浮点数，却被识别成了 `object`（字符串）。这说明这两列里混入了非数值内容。

---

## 2.2 Solidity 的秘密

`Solidity` 是面积与凸包面积的比值，理论范围 0~1。但这里它是 `object` 类型。检查里面有什么：

```python
>>> df['Solidity'].apply(
...     lambda x: x if not str(x).replace('.','').replace('-','').isdigit()
...     else None
... ).dropna()
```

输出结果：

```
?    202    ← 202 个问号！
```

**202 个样本的 Solidity 值是 `?`**。这是 UCI 数据集常见的缺失值占位符。

> 💡 **为什么是 `?`？** UCI 的原始数据格式（.arff）用 `?` 标记缺失值。教师直接从原始文件转换，故意保留了这些标记来模拟真实场景。

这些 `?` 让 pandas 误判整列为字符串。我们需要先把它们转成 `NaN`，再填充。

---

## 2.3 Compactness 带的尾巴

`Compactness` 也是 `object`。看看值长什么样：

```python
>>> df['Compactness'].value_counts().head(8)
0.8194 cm             3    ← 带了 " cm" 后缀！
0.8058 cm             3
0.6921 cm             3
0.8374 cm             2
0.6785015334152743    2    ← 正常值
0.6877689206199188    2
0.6856699544570436    2
0.7167374692135031    2
```

问题很清楚了——部分值的末尾被错误地附上了 ` cm`（注意空格+单位）。这不是全部样本的问题，而是**部分污染**。混合了正常值和带后缀的值，导致整列被识别为字符串。

> 🔍 **诊断技巧**：当数值列显示为 `object` 时，用 `value_counts()` 快速扫一眼，往往能发现污染的元凶。

---

## 2.4 缺失的 Perimeter

接下来检查缺失值：

```python
>>> df.isnull().sum()[df.isnull().sum() > 0]
Perimeter    469    ← 469 个缺失值
Solidity     272    ← 这部分其实也是 "?" 导致的非数值问题
```

`Perimeter` 直接就有 **469 个 NaN**（约 4.9%），分布在三个数据集中：

| 数据集 | 缺失数 | 占比 |
|--------|:---:|------|
| 训练集 | 469 | 4.9% |
| 验证集 | 62 | 4.6% |
| 测试集 | 146 | 5.3% |

三个集的缺失比例大致相同（~5%），说明是**随机缺失**而非系统性缺失。

> 🤔 **为什么是 Perimeter？** 周长（Perimeter）依赖豆粒轮廓的完整提取。图像中如果豆粒边缘模糊或粘连，轮廓提取就会失败。

---

## 2.5 混乱的标签

检查类别标签：

```python
>>> df['Class'].value_counts()
DERMASON     2365
SIRA         1781
SEKER        1343
HOROZ        1270
CALI         1120
BARBUNYA      898
BOMBAY        344
dermason       61    ← 小写！
D3RMAS0N       56    ← 数字替换字母！
sira           37    ← 小写！
...
# 总共 25 个独特值，而不是 7 个！
```

7 个类别，出现了 **25 种变体**。错误模式可以归纳为三类：

**🔡 大小写混乱**（最常见）
```
dermason → DERMASON
sira     → SIRA
horoz    → HOROZ
seker    → SEKER
barbunya → BARBUNYA
cali     → CALI
bombay   → BOMBAY
```

**🔢 数字替换字母**（模拟 OCR 错误）
```
D3RMAS0N → DERMASON   (E→3, O→0)
H0R0Z    → HOROZ      (O→0)
S3K3R    → SEKER      (E→3)
B0MBAY   → BOMBAY     (O→0)
```

**📏 末尾多余空格**
```
"DERMASON " → DERMASON
```

> 💡 这完美模拟了真实世界中数据录入不一致的问题——不同人员、不同系统、不同时期的标注格式可能各不相同。

---

## 本章小结

通过逐步诊断，我们发现了 4 类数据问题：

```
数据质量问题清单
├── Solidity：202 个 "?" 占位符
├── Compactness：部分值含 " cm" 后缀
├── Perimeter：~5% 随机缺失
└── Class：25 种标签变体（应为 7 种）
```

这些问题都不是致命性的，但每一个都需要在建模前妥善处理。下一章将逐一解决它们。

> 📌 **心得**：数据分析不是走流程，而是**带着好奇心去探索**。每一个异常背后都可能藏着有趣的故事——`?` 来自 UCI 的 .arff 格式，` cm` 来自某次数据导出时的格式错误，标签混乱模拟了多源数据合并的典型场景。
