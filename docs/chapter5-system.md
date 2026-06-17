# 5. 系统架构 & 使用说明

## 5.1 项目结构

```
AIT209-DryBean-Classification/
├── data/                        # 数据集（gitignore，不提交）
│   ├── Dry_Bean_Dataset_Dirty_train.csv
│   ├── Dry_Bean_Dataset_Dirty_val.csv
│   └── Dry_Bean_Dataset_Dirty_test.csv
├── docs/                        # GitBook 文档章节
│   ├── chapter1-dataset.md
│   ├── chapter2-eda.md
│   ├── chapter3-preprocessing.md
│   ├── chapter4-experiments.md
│   ├── chapter5-system.md
│   └── chapter6-summary.md
├── src/                         # 源代码
│   ├── data_loader.py           # 数据加载 & 标签映射表
│   ├── preprocess.py            # 预处理流水线（清洗→编码→标准化）
│   ├── run.py                   # 训练 & 评估入口脚本
│   └── train.py                 # 完整流水线（含鲁棒性测试）
├── results/                     # 实验结果
│   └── model_comparison.csv     # 模型对比表
├── models/                      # 训练好的模型（gitignore）
├── SUMMARY.md                   # GitBook 目录
├── README.md                    # 项目首页
└── requirements.txt             # 依赖
```

## 5.2 模块说明

### `data_loader.py`
负责加载三个 CSV 文件，提供 `get_class_mapping()` 函数返回 25→7 的标签映射表。

### `preprocess.py`
核心预处理模块：
- `clean_labels()` — 标签清洗
- `fix_solidity()` — 修复 Solidity 的 `?`
- `fix_compactness()` — 修复 Compactness 的单位后缀
- `impute_missing()` — 中位数填充
- `preprocess_pipeline()` — **对外唯一接口**，返回处理好的数据

### `run.py`
训练和评估入口。定义了 6 个模型的构建函数，统一训练接口，输出完整的对比表格。

## 5.3 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行训练

```bash
# 完整训练 & 评估
cd src
python run.py
```

### 输出示例

```
==========================================================================================
FINAL RESULTS
==========================================================================================
            Model  Train Acc  Val Acc  Test Acc  Test F1  Overfit Gap  Train Time (s)  Inference (ms)
        SVM (RBF)     0.9367   0.9272    0.9331   0.9332       0.0095          0.5802        453.0522
...
Best: SVM (RBF) with Test Acc = 0.9331
```

### 单独运行预处理

```bash
cd src
python preprocess.py
```

## 5.4 设计原则

| 原则 | 实现方式 |
|------|----------|
| **模块化** | 数据加载、预处理、训练分离 |
| **单一入口** | `preprocess_pipeline()` 一个函数完成所有预处理 |
| **防数据泄露** | fit_transform 用在训练集，transform 用在验证/测试集 |
| **可复现** | 所有 random_state 固定为 42 |
| **命令行执行** | 无 UI，纯命令行运行 |
