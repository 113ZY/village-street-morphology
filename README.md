# 村庄道路分析项目说明

代码流程主要分为四步：形态指标聚类、分类模型训练与评估、模型结果可视化、SHAP 解释分析。

## 项目结构

```text
.
├── data/
│   ├── 仅因变量标准化数据.xlsx
│   └── 仅自变量标准化数据.xlsx
├── scripts/
│   ├── run_morphology_clustering.py
│   ├── main.py
│   ├── plot_results.py
│   └── shap_analysis.py
├── src/
│   ├── village_morphology_clustering/
│   │   └── pipeline.py
│   └── models/
│       ├── base.py
│       ├── lr/
│       ├── dt/
│       ├── rf/
│       ├── xgboost/
│       ├── lightgbm/
│       ├── tabm/
│       ├── tabpfn/
│       └── tabicl/
├── xRFM/
├── outputs/
│   ├── clustering/
│   ├── models/
│   ├── figures/
│   └── shap/
└── requirements.txt
```

## 数据说明

项目默认读取 `data/` 目录下的两个 Excel 文件：

- `data/仅因变量标准化数据.xlsx`：用于形态聚类的因变量指标数据。
- `data/仅自变量标准化数据.xlsx`：用于分类模型和 SHAP 分析的自变量指标数据。

形态聚类使用的指标在 `src/village_morphology_clustering/pipeline.py` 中定义：

```python
MORPHOLOGY_COLUMNS = ["MBC", "LCC", "GCC", "INT", "CSR", "IR", "AEL"]
```

分类模型和 SHAP 分析使用的 14 个自变量在 `scripts/main.py` 和 `scripts/shap_analysis.py` 中定义：

```python
X_COLS = [
    "ELEVATION", "SLOPE", "ASPECT", "MAP", "MAT",
    "NDVI", "DWS", "RD", "POP", "PGDP", "NLI",
    "CRPU", "ICHD", "DAR",
]
```

类别标签来自聚类输出文件：

```text
outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx
```

三类在 SHAP 分析中对应为：

- `1 -> LIST`
- `2 -> HIAT`
- `3 -> OECT`

## 环境安装

基础依赖写在 `requirements.txt` 中：

```bash
pip install -r requirements.txt
```

`requirements.txt` 当前包含 `pandas`、`numpy`、`scikit-learn`、`matplotlib`、`openpyxl` 等基础库。项目中还包含 XGBoost、LightGBM、TabM、TabPFN、TabICL、xRFM 等模型入口，如果运行这些模型，需要确保对应包或本地代码可用。

## 总体流程

推荐按以下顺序运行：

```bash
# 1. 形态指标聚类，生成 k2/k3 聚类结果
python scripts/run_morphology_clustering.py

# 2. 使用聚类标签训练并评估模型，推荐 GroupKFold CV
python scripts/main.py --all --cv

# 3. 根据模型输出生成性能表格和图
python scripts/plot_results.py

# 4. 基于 TabICL 做 SHAP 可解释性分析
python scripts/shap_analysis.py
```

## 1. 形态聚类流程

入口脚本：

```bash
python scripts/run_morphology_clustering.py
```

实际执行逻辑在：

```text
src/village_morphology_clustering/pipeline.py
```

主要步骤包括：

1. 从 `data/` 中读取包含 `ID`、`YEAR`、`MBC`、`LCC`、`GCC`、`INT`、`CSR`、`IR`、`AEL` 的 Excel 文件。
2. 提取 `VILLAGE_ID`。
3. 对形态指标进行缺失值处理和 IQR winsorization 异常值缩尾。
4. 使用 `RobustScaler` 标准化。
5. 使用 PCA 保留累计解释方差达到 0.85 的主成分。
6. 使用 KMeans 分别进行 `K=2` 和 `K=3` 聚类。
7. 输出聚类标签、聚类中心、PCA 结果、转移矩阵和诊断结果。

主要输出：

```text
outputs/clustering/k2/
outputs/clustering/k3/
outputs/clustering/k2_k3_clustering_metrics_summary.xlsx
```

其中 `k3` 结果会被后续模型训练默认使用。重要文件包括：

```text
outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx
outputs/clustering/k3/tables/cluster_profiles.xlsx
outputs/clustering/k3/tables/cluster_transition_matrices.xlsx
outputs/clustering/k3/figures/pca_cluster_scatter.png
outputs/clustering/k3/figures/cluster_center_radar.png
outputs/clustering/k3/figures/cluster_transition_heatmaps.png
```

## 2. 模型训练与评估

入口脚本：

```bash
python scripts/main.py --all --cv
```

该脚本读取：

```text
data/仅自变量标准化数据.xlsx
outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx
```

默认支持的模型列表在 `scripts/main.py` 中定义：

```python
ALL_MODELS = ["lr", "dt", "rf", "xgboost", "lightgbm", "tabm", "tabpfn", "tabicl", "xrfm"]
```

可单独运行某个模型，例如：

```bash
python scripts/main.py --lr --cv
python scripts/main.py --rf --cv
python scripts/main.py --tabicl --cv
python scripts/main.py --xrfm --cv
```

也可以进行 xRFM 参数搜索：

```bash
python scripts/main.py --tune-xrfm
```

### 特征工程

`scripts/main.py` 中会基于 14 个原始自变量构造扩展特征。主要包括：

- 原始 14 个自变量。
- 年份归一化特征。
- 年份正弦、余弦时间特征。
- 原始变量与年份的交互项。
- 同一村庄不同年份之间的自变量差分。
- 上一年形态因变量滞后特征。
- 村庄内自变量均值。
- 村庄内自变量标准差。

最终拼接为模型输入矩阵，并在 CV 模式下每折内部单独标准化，避免全局标准化带来的信息泄露。

### 评估方式

推荐使用 `--cv`，代码中采用 `GroupKFold`，分组变量为 `VILLAGE_ID`，用于避免同一村庄的不同年份样本同时出现在训练折和测试折中。

评估指标由 `src/models/base.py` 和 `scripts/main.py` 共同完成，主要包括：

- Accuracy
- Precision macro / weighted
- Recall macro / weighted
- F1 macro / weighted
- AUC OvR
- AUC OvO
- 混淆矩阵

主要输出：

```text
outputs/models/all_models_cv_summary.xlsx
outputs/models/{model}/{model}_cv_results.xlsx
outputs/models/{model}/{model}_test_predictions.xlsx
```

## 3. 模型结果可视化

入口脚本：

```bash
python scripts/plot_results.py
```

该脚本读取：

```text
outputs/models/all_models_cv_summary.xlsx
outputs/models/{model}/{model}_cv_results.xlsx
outputs/models/{model}/{model}_test_predictions.xlsx
```

并输出到：

```text
outputs/figures/
```

主要生成内容：

```text
Table1_train_performance.xlsx
Table2_test_performance.xlsx
Table3_tabicl_per_class.xlsx
Fig1_test_heatmap.png
Fig2_roc_curves.png
Fig3_auc_forest.png
Fig4_radar.png
Fig5_confusion_matrices.png
Fig_cm_{model}.png
```

其中混淆矩阵已经按三分类绘制，类别为 `Cluster 1`、`Cluster 2`、`Cluster 3`。

## 4. SHAP 可解释性分析

入口脚本：

```bash
python scripts/shap_analysis.py
```

当前 SHAP 分析脚本使用的是 TabICL 模型。主要流程为：

1. 读取 `data/仅自变量标准化数据.xlsx`。
2. 读取 `outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx`。
3. 构造与模型训练类似的扩展特征。
4. 训练 TabICL 分类模型。
5. 基于原始 14 个自变量构造 SHAP 预测函数。
6. 使用 `shap.KernelExplainer` 计算三分类 SHAP 值。
7. 将 SHAP 值保存到文件，后续可直接加载，避免重复计算。
8. 分别为 `LIST`、`HIAT`、`OECT` 三类输出 SHAP 图。

SHAP 值保存路径格式为：

```text
outputs/shap/shared/tabicl_shap_values_n{n_explain}_bg{n_background}_ns100.npz
```

每个类别的图输出到：

```text
outputs/shap/LIST/
outputs/shap/HIAT/
outputs/shap/OECT/
```

主要图件包括：

```text
Fig2_shap_summary_{class}.png
Fig3_dependence_{class}.png
Fig4_main_vs_interaction_{class}.png
Fig5_interaction_matrix_{class}.png
Fig6_bivariate_{featureA}x{featureB}_{class}.png
Fig6_bivariate_summary_{class}.png
Fig7_heatmap_{class}.png
Fig8_waterfall_{class}.png
```

其中：

- `Fig2`：SHAP 全局重要性与分布图。
- `Fig3`：单特征 SHAP 依赖图。
- `Fig4`：主效应与二阶交互效应对比。
- `Fig5`：二阶扰动交互矩阵。
- `Fig6`：双因子交互图，包括单独交互图和汇总图。
- `Fig7`：SHAP 热力图。
- `Fig8`：个体样本 SHAP force plot。

当前代码中的交互分析使用二阶扰动交互思路，核心函数包括：

```text
compute_second_order_interactions()
compute_pair_second_order_values()
```

预设关注的交互对在 `INTERACTION_PAIRS` 中定义：

```python
INTERACTION_PAIRS = [
    ("DAR", "NLI"),
    ("DAR", "RD"),
    [("SLOPE", "NLI"), ("SLOPE", "PGDP"), ("ELEVATION", "NLI"), ("ELEVATION", "PGDP")],
    [("MAP", "NDVI"), ("MAT", "NDVI")],
    ("DAR", "CRPU"),
    ("CRPU", "ICHD"),
]
```

对于列表形式的候选交互项，代码会根据交互强度选择其中效果较好的组合进行绘图。

## 常用命令

```bash
# 运行聚类
python scripts/run_morphology_clustering.py

# 所有模型 GroupKFold CV
python scripts/main.py --all --cv

# 单独运行 TabICL
python scripts/main.py --tabicl --cv

# 单独运行 xRFM
python scripts/main.py --xrfm --cv

# xRFM 参数搜索
python scripts/main.py --tune-xrfm

# 绘制模型性能图
python scripts/plot_results.py

# 绘制 SHAP 解释图
python scripts/shap_analysis.py
```

## 输出目录说明

```text
outputs/clustering/
```

保存形态聚类相关表格、诊断文件和聚类图。

```text
outputs/models/
```

保存各模型的 CV 结果、测试预测结果和模型汇总表。

```text
outputs/figures/
```

保存模型性能对比图、ROC 曲线、AUC 森林图、雷达图和混淆矩阵。

```text
outputs/shap/
```

保存 TabICL 的 SHAP 值文件和三类村庄对应的 SHAP 解释图。

## 注意事项

1. `scripts/main.py` 默认使用 `outputs/clustering/k3/tables/village_year_cluster_assignments.xlsx` 作为分类标签，因此应先运行聚类脚本。
2. `scripts/plot_results.py` 依赖 `outputs/models/all_models_cv_summary.xlsx` 和各模型预测文件，因此应先运行模型训练脚本。
3. `scripts/shap_analysis.py` 计算 SHAP 较慢，代码已将 SHAP 值保存为 `.npz` 文件；同样参数下再次运行会优先加载已有文件。
4. `requirements.txt` 只包含基础依赖，运行 XGBoost、LightGBM、TabM、TabPFN、TabICL 或 xRFM 时，需要确保这些模型依赖已经安装或本地模块路径可用。
5. 项目中部分绘图脚本使用 `matplotlib.use("Agg")`，适合无图形界面的服务器环境，图片会直接保存到 `outputs/` 下。

