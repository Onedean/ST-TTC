# Spatio-temporal OOD Learning Setting

_In this sub-setting, we use and adapt the [code repository](https://github.com/PoorOtterBob/STONE-KDD-2024) of the **STONE** model._

## 1. Generating the SD and GBA sub-datasets from CA dataset
In the experiments of our paper, we used SD dataset with years from 2019 to 2020, which were generated from CA dataset, followed by [LargeST](https://github.com/liuxu77/LargeST/blob/main). For example, you can download CA dataset from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest) and please place the downloaded `archive.zip` file in the `data/ca` folder and unzip the file. 

First of all, you should go through a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Then, please go through all the cells in the provided jupyter notebooks `generate_sd_dataset.ipynb` in the folder `data/sd` and `generate_gla_dataset.ipynb` in the folder `data/gla` respectively. Finally use the commands below to generate traffic flow data for our experiments. 
```
python data/generate_data_for_training.py --dataset sd_gba --years 2019
python data/generate_data_for_training.py --dataset sd_gba --years 2020
```

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [LargeST](https://github.com/liuxu77/LargeST/blob/main).

<br>

## 3. Model Running
To train STONE on <b>SD</b>, for example, you may execute this command in the terminal:
```
bash experiments/stone/run_train.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 0 --bs 64 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1
```

---

To test STONE w/o ST-TTC (i.e., Normal test) on <b>SD</b>, 
First, you should make sure that **line 231** in [main.py](./experiments/stone/main.py) file contains the following:
```
engine.evaluate(args.mode)
```
and then, you may execute this command in the terminal:
```
bash experiments/stone/run_test.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 0 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1 --mode test
```

---

To test STONE w/ ST-TTC on <b>SD</b>, 
First, you should make sure that **line 231** in [main.py](./experiments/stone/main.py) file contains the following:
```
engine.evaluate_with_ttc(args.mode, args.group, args.sd_lr)
```
and then, you may execute this command in the terminal:
```
bash experiments/stone/run_test.sh
```
or directly execute the Python file in the terminal:
```
python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 0 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1 --mode test
```

