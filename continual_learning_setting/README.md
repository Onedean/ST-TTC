# Spatio-temporal Continual Learning Setting

_In this sub-setting, we use and adapt the [code repository](https://github.com/Onedean/EAC) of the **EAC** framework._


## 📚 Training Data

The processed dataset can be directly accessed from the [cloud disk](https://hkustgz-my.sharepoint.com/:f:/g/personal/wchen110_connect_hkust-gz_edu_cn/EuiKtt95qnpNgOngXAV_MmABWYyEBh74ooM94kdycwg4Sw?e=ZRCC1n)!

Please download all processed datasets and place them in the [data folder](./data).

## 🚀 Getting Started

### Installation

1. Please install the core dependencies, including:

```shell
python = 3.8.5
pytorch = 1.7.1
torch-geometric = 1.6.3
```

2. Or you can directly create and import a ready-made environment:

```shell
conda env create -f environment.yaml
conda activate stg
```

### Usages

Before you get started, you need to make sure your data and trained weights are ready!

---
To test EAC or STKEC w/o ST-TTC (i.e., Normal test) on different datasets:

First, you should make sure that **line 60** & **line 153** in [main.py](./main.py) file and **line 56** & **line 148** in [stkec_main.py](./stkec_main.py) contains the following:
```
test_model(model, args, test_loader, pin_memory=True)
```
and then, your can run a specific method on a specific dataset separately:
```python
python main.py --conf conf/PEMS/eac.json --gpuid 0 --seed 43
python stkec_main.py --conf conf/PEMS/eac.json --gpuid 0 --seed 43
```
Or you can run the script to batch execute all baseline methods on a specified dataset, for example, run all baseline methods on the PEMS-Stream dataset:

```shell
sh scripts/pems_run.sh
```

---
To test EAC or STKEC w/ ST-TTC on different datasets:

First, you should make sure that **line 60** & **line 153** in [main.py](./main.py) file and **line 56** & **line 148** in [stkec_main.py](./stkec_main.py) contains the following:
```
test_model_with_ttc(model, args, test_loader, pin_memory=True)
```
and then, your can run a specific method on a specific dataset separately:
```python
python main.py --conf conf/PEMS/eac.json --gpuid 0 --seed 43
python stkec_main.py --conf conf/PEMS/eac.json --gpuid 0 --seed 43
```
Or you can run the script to batch execute all baseline methods on a specified dataset, for example, run all baseline methods on the PEMS-Stream dataset:

```shell
sh scripts/pems_run.sh
```