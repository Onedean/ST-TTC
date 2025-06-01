# Small Scale Spatio-Temporal Forecasting Scenario

_In this sub-scenario, we use and adapt the [code repository](https://github.com/liuxu77/LargeST) of the **LargeST** repository._

## 1. Data Preparation

Follow the **LargeST** data format and handle all the benchmark datasets you want to use.


## 2. Experiments Running

To test Model (e.g. GWNet) w/o ST-TTC (i.e., Normal test) on different datasets (e.g. PEMS-08), you may execute the Python file in the terminal:
```
python experiments/gwnet/main.py --device cuda:1 --dataset PEMS08 --fewshot 100 --years 2016 --model_name gwnet --seed 2017 --bs 1 --mode test --eval_method norm
```
---
To test Model (e.g. GWNet) w/ ST-TTC on different datasets (e.g. PEMS-08), you may execute the Python file in the terminal:
```
python experiments/gwnet/main.py --device cuda:1 --dataset PEMS08 --fewshot 100 --years 2016 --model_name gwnet --seed 2017 --bs 1 --mode test --eval_method ttc
```
