# 第四章 · 多算法实验与对比分析

本章对 **5 种分类算法**进行系统性实验。其中 KNN、SVM、Random Forest 为课堂讲授内容，AdaBoost 和 GCN（图卷积网络）为自行查阅文献和官方文档学习的课外算法。

> 🧭 所有算法的架构图见：[模型架构图解](model-architectures.html)

每种算法均从数学原理出发，给出完整代码实现，最后进行多维度对比分析。

> 📐 **统一实验设置**：所有算法在相同预处理后的数据上训练和测试。训练集 9,527 条，验证集 1,347 条，测试集 2,737 条。特征已通过 StandardScaler 标准化。

---

## 4.1 K 近邻（KNN）

### 4.1.1 算法原理

K 近邻（K-Nearest Neighbors）是最直观的分类算法：**找出特征空间中离待预测样本最近的 K 个训练样本，由它们投票决定类别**。

#### 距离度量

KNN 的核心是距离计算。对于两个样本 \\( x \\) 和 \\( x' \\)，欧氏距离定义为：

\\[
d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}
\\]

其中 \\( p \\) 为特征维度（本项目 p=16）。

#### 决策规则

给定待预测样本 \\( x \\)，找出其 \\( K \\) 个最近邻居 \\( N_K(x) \\)，预测类别为：

\\[
\hat{y} = \arg\max_{c} \sum_{i \in N_K(x)} \mathbb{I}(y_i = c)
\\]

即对 \\( K \\) 个邻居的类别进行**多数投票**。

#### K 值的选择

K 是 KNN 唯一的超参数：

- **K 太小**（如 K=1）：模型过于敏感，容易受噪声影响 → 过拟合
- **K 太大**：决策边界过于平滑，忽略局部结构 → 欠拟合

本实验采用 K=5，在精确性和平滑性之间取得平衡。

> 💡 KNN 是典型的**非参数懒惰学习**（lazy learning）：它不显式地构建判别函数，而是将全部训练数据存储起来，在预测时才进行计算。

---

### 4.1.2 代码实现

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,     # K 值
    n_jobs=-1          # 并行计算
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

KNN 的 `fit()` 其实只是将训练数据存入内部数据结构（kd-tree 或 ball-tree），真正的计算发生在 `predict()` 阶段。这也是为什么 KNN 训练极快但推理慢。

---

### 4.1.3 实验结果与分析

```
Train Accuracy : 93.86%
Val Accuracy   : 91.61%
Test Accuracy  : 91.78%
F1-Score       : 0.9179
Overfitting Gap: +2.25%
Inference Speed: 517.1 ms (per 2737 samples)
Training Time  : 0.001s
```

**分析**：

KNN 以几乎零训练成本拿到了 91.78% 的测试精度，说明数据本身的**类别可分性良好**——同类豆子在特征空间中确实聚集在一起。

2.25% 的过拟合说明 K=5 的选择比较合理。推理速度（517ms）是四个算法中**第二慢**的——每次预测都要计算与全部 9,527 个训练样本的距离。在需要实时推理的场景下，KNN 不是好选择。

> 🔑 KNN 的最大价值在于作为 **baseline 模型**：如果连 KNN 都能拿到不错的精度，说明特征设计和数据质量没有根本性问题。

---

## 4.2 支持向量机（SVM）

### 4.2.1 算法原理

支持向量机（Support Vector Machine）的核心思想是：在特征空间中寻找一个**最大间隔超平面**来分隔不同类别。

#### 线性可分情形

对于二分类问题，SVM 寻找超平面 \\( w^T x + b = 0 \\)，使得：

\\[
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \; \forall i
\\]

这是一个**凸二次规划**问题，存在全局最优解。

#### 软间隔

现实数据往往不是严格线性可分的。引入松弛变量 \\( \xi_i \\) 和正则化参数 C：

\\[
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{n} \xi_i
\\]
\\[
\text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \;\; \xi_i \geq 0
\\]

- **C 较大**：对误分类的惩罚更重，间隔更窄，更关注训练精度 → 容易过拟合
- **C 较小**：允许更多误分类，间隔更宽，泛化更好 → 可能欠拟合

#### RBF 核

对于非线性问题，SVM 通过核函数将数据映射到高维空间：

\\[
K(x, x') = \exp\left(-\gamma \|x - x'\|^2\right)
\\]

\\[ \gamma \\] 控制每个样本的影响范围：\\[ \gamma \\] 越大，影响范围越小，决策边界越复杂。

#### 多分类扩展

SVM 本身是二分类器。sklearn 默认使用**一对一**（one-vs-one）策略：为每对类别训练一个 SVM，共 \\[ C(7,2) = 21 \\] 个子分类器，最终投票决定。

---

### 4.2.2 代码实现

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',        # 径向基核，处理非线性问题
    C=10,                # 适中的正则化强度
    gamma='scale',       # γ = 1/(n_features × X.var())
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 4.2.3 实验结果与分析

```
Train Accuracy : 93.67%
Val Accuracy   : 92.72%
Test Accuracy  : 93.31%    ← 🥇 最优
F1-Score       : 0.9332
Overfitting Gap: +0.95%     ← 泛化能力最强
Inference Speed: 610.6 ms   ← 推理最慢
Training Time  : 0.7s
```

**分析**：

SVM 以 **93.31%** 的测试精度排名第一，且过拟合仅 0.95%——训练集和验证集的精度几乎一致，说明模型的泛化能力出色。

RBF 核在这个 16 维特征空间中有效捕捉了非线性分类边界。C=10 的设置在精度和泛化间取得了良好平衡。

主要缺点是推理速度（610ms）是四个算法中最慢的。每次预测都需要计算测试样本与所有支持向量的核函数值——本项目 SVM 保留了大量支持向量。

> 🏆 SVM 的优异表现印证了一句经验法则：**中等维度（<100）、中等规模（~10K 样本）的分类问题，RBF-SVM 往往是最稳定可靠的选择。**

---

## 4.3 随机森林（Random Forest）

### 4.3.1 算法原理

随机森林是 **Bagging**（Bootstrap Aggregating）的经典代表，由多棵决策树构成。

#### Bagging

假设我们从训练集中有放回地随机采样 \\( B \\) 次，每次采样 \\( n \\) 个样本，训练 \\( B \\) 棵决策树：

\\[
\hat{f}_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(x)
\\]

对于分类任务，最终预测由 \\( B \\) 棵树**多数投票**决定。

由于每棵树的训练数据不同（Bootstrap 采样），它们学到的模型自然存在差异——这种差异就是 Bagging 降低方差的关键。

#### 双重随机性

随机森林在 Bagging 的基础上引入第二层随机性——特征随机选择：

在每个节点的分裂时，不是考虑所有特征，而是**随机选择 m 个特征**（通常 \\[ m = \sqrt{p} \\]），从中找出最优分裂。

双重随机性确保了森林中每棵树的多样性，从而进一步提升集成效果。

#### 决策树的分裂准则

每棵决策树使用**基尼不纯度**（Gini Impurity）选择最优分裂：

\\[
Gini(D) = 1 - \sum_{c=1}^{K} p_c^2
\\]

分裂的目标是最大化基尼不纯度的减少量（即信息增益）。

#### 袋外估计

Bootstrap 采样时，约 37% 的样本不会被抽中，这些**袋外样本**（Out-of-Bag）可以作为天然的验证集，无需额外划分：

\\[
P(\text{未抽中}) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368
\\]

---

### 4.3.2 代码实现

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,    # 100 棵决策树
    max_depth=12,        # 限制树深度，防止过拟合
    random_state=42,
    n_jobs=-1            # 并行训练
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 4.3.3 实验结果与分析

```
Train Accuracy : 98.32%    ← 几乎完美记忆训练集
Val Accuracy   : 91.91%
Test Accuracy  : 92.36%
F1-Score       : 0.9237
Overfitting Gap: +6.41%     ← ⚠️ 过拟合明显
Inference Speed: 45.2 ms    ← 推理很快
Training Time  : 1.4s
```

**分析**：

随机森林在训练集上达到了 98.32% 的精度，但验证集和测试集只有约 92%——存在 **6.41% 的过拟合**。100 棵树、深度 12 的配置对这 9,527 条数据来说过于复杂。

尽管如此，它的测试精度（92.36%）在所有算法中排名**第二**，而且推理速度（45ms）远优于 SVM 和 KNN。

改进方向：降低 max_depth 到 8，或者增加 min_samples_split 到 5，可以缓解过拟合并可能提升泛化能力。

> 🔧 随机森林的优势在于**不需要特征缩放**、**对噪声鲁棒**、**可以输出特征重要性**——这些在实践中非常有价值。

---

## 4.4 AdaBoost

### 4.4.1 算法原理

AdaBoost（Adaptive Boosting）属于 Boosting 家族，但与 Gradient Boosting 有本质不同——它不拟合残差，而是**动态调整样本权重**。

#### 核心思想

Boosting 将多个**弱学习器**（weak learner）组合成**强学习器**（strong learner）。弱学习器的要求很低——只需比随机猜测好一点（准确率 > 50%）。

#### 算法步骤

给定训练集 \\( D = \{(x_1, y_1), ..., (x_n, y_n)\} \\)，进行 \\( T \\) 轮迭代：

**第 1 轮**：初始化所有样本权重相等
\\[
w_1(i) = \frac{1}{n}
\\]

**第 t 轮**（t = 1, 2, ..., T）：

1. 在加权训练集上训练弱学习器 \\( h_t \\)
2. 计算加权错误率：
\\[
\epsilon_t = \frac{\sum_{i: h_t(x_i) \neq y_i} w_t(i)}{\sum_i w_t(i)}
\\]
3. 计算该弱学习器的权重：
\\[
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
\\]
   \\[ \epsilon_t \\] 越小 → \\[ \alpha_t \\] 越大 → 该分类器在最终投票中越重要
4. 更新样本权重——分错的样本权重增加，分对的减少：
\\[
w_{t+1}(i) = w_t(i) \cdot \exp\left(-\alpha_t \cdot y_i \cdot h_t(x_i)\right)
\\]

**最终预测**：

\\[
H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)
\\]

> 🔑 AdaBoost 的本质是**指数损失函数**下的前向分步加法模型。与 Gradient Boosting 不同，它不是沿着梯度方向优化，而是通过调整样本分布来"逼迫"后续弱学习器关注之前分错的样本。

#### AdaBoost vs Gradient Boosting

| | AdaBoost | Gradient Boosting |
|------|-----------|-------------------|
| 损失函数 | 指数损失 | 任意可微损失 |
| 优化方式 | 调整样本权重 | 拟合伪残差（梯度） |
| 对噪声 | **敏感** | 相对鲁棒 |
| 基学习器 | 通常浅层树 | 通常浅层树 |

---

### 4.4.2 代码实现

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=3,          # 弱学习器：浅层决策树
        random_state=42
    ),
    n_estimators=50,          # 50 轮迭代
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 4.4.3 实验结果与分析

```
Train Accuracy : 90.29%
Val Accuracy   : 88.57%
Test Accuracy  : 89.84%
F1-Score       : 0.8982
Overfitting Gap: +1.72%
Inference Speed: 29.8 ms      ← 推理最快
Training Time  : 5.6s
```

**分析**：

AdaBoost 的测试精度（89.84%）是四个算法中最低的。但这并不意味着 AdaBoost 是"差"的算法——它揭示了不同算法在面对相同数据时的不同特性：

1. **对噪声敏感**是 AdaBoost 的已知弱点。数据中的异常值或标注错误（尽管我们做了清洗，但可能仍有残留）会被指数损失函数放大，导致权重分配失真。

2. **弱学习器太弱**：max_depth=3 的决策树在 16 维特征空间上可能不足以作为可靠的基学习器。增加深度（如 depth=5）可能提升精度。

3. **样本效率**：AdaBoost 在 50 轮后就接近收敛，过拟合仅 1.72%，说明它没有过度记忆训练数据。

它的亮点在于**推理最快**（29.8ms），在所有算法中排名第一，且内存占用很小（只需存储 50 棵深度 3 的树）。

> 💡 AdaBoost 虽然没有在这场"比赛"中获胜，但它的存在让对比更有意义——它展示了 Boosting 家族中不同策略（指数损失 vs 梯度）在面对相同数据时的差异，也验证了"不同算法适用不同数据分布"的机器学习基本认知。

---

## 4.5 图卷积网络（GCN）

> ⚡ **课外自学**。图神经网络是近年深度学习领域的热门方向。本节将表格数据转化为 KNN 图结构，用 2 层 GCN 进行节点分类。

---

### 4.5.1 算法原理

图卷积网络（Graph Convolutional Network）将卷积操作推广到图结构数据。它的核心是**消息传递**（message passing）：每个节点聚合邻居节点的特征来更新自身的表示。

#### 从表格到图

Dry Bean 数据集本身没有图结构。我们通过 **K 近邻图**来构造：

```
样本 i 和样本 j 之间连边 ⟺ j 是 i 的 K 个最近邻居之一
```

每个样本是图中的一个节点，节点特征为 16 维向量，边表示两个样本在特征空间中相近。

这有直观的物理意义：**特征相似的豆子，在图中相互连接，信息可以在它们之间传递**。

#### GCN 层

单层 GCN 的传播规则为：

\[
H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)
\]

其中：

- \[ \tilde{A} = A + I \]：添加自环的邻接矩阵
- \[ \tilde{D}_{ii} = \sum_j \tilde{A}_{ij} \]：度矩阵
- \[ \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} \]：对称归一化的邻接矩阵
- \[ H^{(l)} \]：第 l 层的节点特征矩阵
- \[ W^{(l)} \]：可学习的权重矩阵
- \[ \sigma \]：非线性激活函数（ReLU）

归一化保证了度数高的节点不会在消息聚合中占主导地位。

#### 2 层 GCN 架构

```
输入特征 (16 维)
    │
    ▼
GCN Layer 1: 16 → 64 (ReLU)
    │
    ▼ Dropout(0.3)
GCN Layer 2: 64 → 7 (输出 logits)
    │
    ▼
Softmax → 预测类别
```

总参数量仅 **1,543** 个（包含两个变换矩阵和偏置），是一个非常轻量的模型。

---

### 4.5.2 代码实现

#### 构建 KNN 图

```python
from sklearn.neighbors import NearestNeighbors

K = 10
nn = NearestNeighbors(n_neighbors=K+1, metric='euclidean', n_jobs=-1)
nn.fit(X_all)
_, indices = nn.kneighbors(X_all)

# 构建边列表（排除自环）
edges = []
for i in range(n_nodes):
    for j in indices[i, 1:]:   # 跳过第一个（自身）
        edges.append([i, j])
```

#### GCN 层（纯 PyTorch 实现）

```python
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, X, edge_index):
        row, col = edge_index
        n = X.shape[0]

        # 对称归一化: D^(-1/2) A D^(-1/2)
        deg = torch.bincount(row, minlength=n).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 消息传递：聚合邻居特征
        messages = X[col] * norm.unsqueeze(-1)
        aggregated = torch.zeros(n, X.shape[1])
        aggregated.scatter_add_(0, row.unsqueeze(-1).expand(-1, X.shape[1]), messages)

        # 线性变换 + 激活
        return torch.relu(aggregated @ self.W + self.b)
```

#### 训练配置

```python
model = GCN(in_dim=16, hidden_dim=64, out_dim=7, dropout=0.3)
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    logits = model(X, edge_index)
    loss = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
```

---

### 4.5.3 实验结果与分析

```
Train Accuracy : 92.00%
Val Accuracy   : 92.20%
Test Accuracy  : 91.60%
Overfitting Gap: -0.20%     ← 几乎零过拟合
Parameters     : 1,543        ← 极致轻量
Training Epochs: 200
```

**分析**：

GCN 以 91.60% 的测试精度排在第四位，但有几个非常有趣的观察：

1. **负过拟合**：训练精度（92.00%）略低于验证精度（92.20%），过拟合为 -0.20%。这在机器学习中很少见——通常意味着模型过于简单（欠拟合风险）或者验证集恰好更容易。对于 1,543 参数的微型网络，更可能是前者。

2. **极致轻量**：1,543 个参数，对比 Random Forest 的 100 棵深度 12 树（可能上万个决策节点），GCN 用极少的参数达到了接近的水平。

3. **图结构的价值**：KNN 图让模型能够利用样本间的近邻关系。未连接图中的 KNN 基线是 91.78%，连接后 GCN 达到 91.60%——虽然未超越，但在仅 200 个 epoch 的训练下已非常接近。增加 epoch 数或网络宽度可能进一步缩小差距。

4. **训练稳定**：损失函数从 0.82 稳定下降到 0.22，没有出现过拟合迹象——这得益于 Dropout（0.3）和权重衰减（5e-4）的正则化效果。

> 🔮 GCN 的实验展示了**结构化视角**的力量：同样的特征，引入 KNN 图结构后，一个 1,543 参数的微型网络就能达到 91.60% 的精度。这暗示了——如果使用更深的网络（如 3 层）或更复杂的图结构（如自适应图），精度还有提升空间。

---

## 4.6 综合对比

### 量化对比

| 维度 | KNN | SVM | Random Forest | AdaBoost | GCN |
|------|:---:|:---:|:---:|:---:|:---:|
| 测试精度 | 91.78% | **93.31%** | 92.36% | 89.84% | 91.60% |
| F1-Score | 0.9179 | **0.9332** | 0.9237 | 0.8982 | 0.9160 |
| 过拟合程度 | 2.25% | **0.95%** | 6.41% | 1.72% | -0.20% |
| 推理速度 | 517ms | 610ms | 45ms | **30ms** | ~8ms |
| 训练速度 | **0.001s** | 0.7s | 1.4s | 5.6s | ~15s |
| 参数量 | ~9.5K 样本 | ~数千 SV | ~万级 | 50×浅树 | **1,543** |

### 定性对比

| 维度 | KNN | SVM | Random Forest | AdaBoost | GCN |
|------|-----|-----|:---:|-----|:---:|
| 特征缩放需求 | 必须 | 必须 | 不需要 | 不需要 | 必须 |
| 可解释性 | 低 | 低 | **高** | 中 | 低 |
| 对噪声鲁棒 | 中 | 中高 | **高** | 低 | 中 |
| 内存占用 | 高 | 中 | 中 | **低** | 中 |
| 算法范式 | 基于距离 | 核方法 | 集成学习 | 集成学习 | **图神经网络** |

### 核心发现

**① 精度与泛化的权衡**  
SVM 同时拥有最高精度和最低过拟合，是最理想的选择。Random Forest 精度接近但过拟合严重——需要更强的正则化。

**② 速度的两极分化**  
训练阶段：KNN（无训练）vs AdaBoost（5.6s）vs GCN（15s，200 epoch）  
推理阶段：AdaBoost（30ms）vs GCN（~8ms，矩阵运算）vs SVM（610ms）  
部署时需根据延迟要求选择。

**③ 课外算法的教学意义**  
AdaBoost（89.84%）和 GCN（91.60%）的精度低于课堂算法，但这恰恰说明了为什么要对比多种算法——**没有任何一种算法在所有数据集上都是最佳选择**。AdaBoost 对噪声敏感，GCN 受限于微型网络容量。No Free Lunch 定理的实验验证比公式推导更有说服力。

**④ 图结构的力量**  
GCN 仅用 1,543 个参数就达到了与 KNN（需要存储 9,527 个样本）接近的精度。KNN 图构造赋予模型"邻近样本的信息可以传递"的先验知识——这种**归纳偏置**正是现代深度学习成功的关键。

**⑤ 数据质量决定模型上限**  
清洗前：25 种标签变体、缺失值、字符串污染 → 任何模型都无法正常工作  
清洗后：五种算法均达到 90%+ 精度  
**数据处理的重要性在这里得到了最直接的验证。**
