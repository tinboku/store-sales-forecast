# Store Sales Forecasting

用 SARIMA、Prophet、LSTM 三种方法预测 Superstore 月度零售销售额，对比统计模型和深度学习在小数据集上的表现。

数据来自 [Kaggle Superstore Sales](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting-dataset)，约 10K 条订单记录（2015-2018），3 个品类，4 个区域。

## 怎么跑

```bash
pip install -r requirements.txt
# 把 kaggle 下载的 CSV 放到 data/raw/train.csv
python run_experiment.py
```

也可以直接看 notebooks：
- `01_eda.ipynb` — EDA 和平稳性检验
- `02_baseline_models.ipynb` — SARIMA + Prophet
- `03_deep_learning.ipynb` — LSTM 实验

跑测试：
```bash
pytest tests/ -v
```

## 结果

训练集 2015-2017，测试集 2018（12 个月）

| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| Seasonal Naive | 18,986 | 15,468 | 24.5% |
| SARIMA(1,1,0)(0,1,1,12) | 21,992 | 18,947 | 44.8% |
| Prophet | 26,629 | 19,075 | 30.9% |
| LSTM | 29,578 | 23,403 | 37.8% |

Seasonal Naive 反而最好 —— 数据只有 36 个月，季节性又很强，直接用去年同月的值就已经很准了。LSTM 在这种数据量下没什么优势，递归预测的误差累积也是个问题。

## 项目结构

```
src/
  data_loader.py      # 数据加载和聚合
  features.py         # 特征工程（lag、rolling、时间特征）
  evaluate.py         # RMSE / MAE / MAPE
  utils.py            # 画图和配置
  models/             # SARIMA / Prophet / LSTM
notebooks/            # EDA + 建模实验
tests/                # 单元测试
configs/config.yaml   # 超参数
results/              # 输出图表（gitignored）
```

## 依赖

Python 3.9+，详见 `requirements.txt`
