# 文本数据管理与分析 - 期末作业

## 作业要求

完成两项文本处理任务：
1. **文本分类**：使用传统机器学习方法（随机森林、SVM、逻辑回归、朴素贝叶斯）
2. **文本聚类**：使用K-means和DBSCAN算法

## 项目结构

```
.
├── main.py              # 主程序：基础实验（文本分类、聚类和深度学习）
├── experiments.py       # 扩展实验：参数调优、不同文本表示方法
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明文档
```

## 环境要求

- Python 3.7+
- 依赖包见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行基础实验

```bash
python main.py
```

这将运行：
- **文本分类实验（传统机器学习）**：比较随机森林、SVM、逻辑回归、朴素贝叶斯四种算法
- **文本聚类实验**：比较K-means和DBSCAN两种聚类算法

### 2. 运行扩展实验

```bash
python experiments.py
```

这将运行：
- 不同文本向量化方法比较（TF-IDF的不同参数配置）
- 是否使用降维的效果比较
- 参数调优实验（SVM、随机森林）
- 聚类算法参数调优

## 数据集

本项目使用 **20 Newsgroups** 英文数据集，包含4个类别：
- alt.atheism
- soc.religion.christian
- comp.graphics
- sci.med

数据集会在首次运行时自动下载。

## 实验结果

### 文本分类

#### 传统机器学习方法

使用TF-IDF向量化，比较四种分类算法：
- 随机森林 (Random Forest)
- 支持向量机 (SVM)
- 逻辑回归 (Logistic Regression)
- 朴素贝叶斯 (Naive Bayes)

评估指标：准确率 (Accuracy)

### 文本聚类

使用TF-IDF向量化，比较两种聚类算法：
- K-means聚类
- DBSCAN聚类

评估指标：
- 调整兰德指数 (ARI - Adjusted Rand Index)
- 轮廓系数 (Silhouette Score)

### 扩展实验

1. **不同向量化方法**：
   - TF-IDF的不同n-gram范围（1-gram, 2-gram, 1-2gram）
   - 不同的特征数量（1000, 3000, 5000, 10000）

2. **特征降维**：
   - 无降维
   - PCA降维到不同维度（100, 300, 500）

3. **参数调优**：
   - SVM：C、kernel、gamma参数网格搜索
   - 随机森林：n_estimators、max_depth、min_samples_split参数调优
   - K-means：不同聚类数量（2, 3, 4, 5, 6）
   - DBSCAN：不同eps和min_samples参数

## 代码说明

### main.py

主程序包含核心函数：
- `experiment_text_classification()`: 文本分类实验（传统机器学习）
- `experiment_text_clustering()`: 文本聚类实验

### experiments.py

扩展实验程序包含：
- `experiment_different_vectorizations()`: 不同向量化方法实验
- `experiment_parameter_tuning()`: 参数调优实验
- `experiment_clustering_parameters()`: 聚类参数调优
- `plot_results()`: 结果可视化
