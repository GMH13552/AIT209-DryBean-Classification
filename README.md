# 🫘 Dry Bean 多分类 — 机器学习全流程项目

> **AIT209 机器学习与项目实践 · 期末作业**  
> 数据集：Dry Bean Dataset（UCI, 2020）  
> 任务：7 种干豆的分类识别  

---

## 关于本项目

本项目是一个完整的机器学习工程项目，包含**总结论文**和**系统实现**两部分。

**总结论文**按电子书组织，涵盖数据分析、数据预处理、多算法实验和课程总结五大模块。

**系统实现**采用模块化 Python 工程，由 `main.py` 统一调度，做到一键训练、一键评估。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流水线
py src/main.py
```

## 目录导航

### 第一部分：总结论文

| 章节 | 内容 |
|------|------|
| [第一章](docs/chapter1-dataset.md) | 数据集概览 |
| [第二章](docs/chapter2-eda.md) | 数据质量分析 |
| [第三章](docs/chapter3-preprocessing.md) | 数据预处理 |
| [第四章](docs/chapter4-experiments.md) | 多算法实验 |
| [第五章](docs/chapter5-summary.md) | 课程总结 |

### 第二部分：系统展示

| 章节 | 内容 |
|------|------|
| [第六章](docs/chapter6-system.md) | 系统架构 |

---

## 算法与结果

| 排名 | 算法 | 测试精度 | 来源 |
|:---:|------|:---:|:---:|
| 🥇 | SVM (RBF) | **93.31%** | 课堂 |
| 🥈 | Random Forest | 92.36% | 课堂 |
| 🥉 | KNN | 91.78% | 课堂 |
| 4 | AdaBoost | 89.84% | 课外 ✨ |

---

## 技术栈

`Python` `scikit-learn` `pandas` `numpy` `GitBook`

---

*最后更新：2026-06-17*
