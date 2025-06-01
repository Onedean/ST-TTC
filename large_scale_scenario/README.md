# Large Scale Spatio-Temporal Forecasting Scenario

_In this sub-scenario, we use and adapt the [code repository](https://github.com/PoorOtterBob/STONE-KDD-2024) of the **PatchSTG** model._


## Requirements
- torch==1.11.0
- timm==1.0.12
- scikit_learn==1.0.2
- tqdm==4.67.1
- pandas==1.4.1
- numpy==1.22.3

## Folder Structure

```tex
└── code-and-data
    ├── config                 # Including detail configurations
    ├── cpt                    # Storing pre-trained weight files (manually create the folder and download files)
    ├── data                   # Including traffic data (download), adj files (generated), and the meta data
    ├── lib
    │   |──  utils.py          # Codes of preprocessing datasets and calculating metrics
    ├── log                    # Storing log files
    ├── model
    │   |──  models.py         # The core source code of our PatchSTG
    ├── main.py                # This is the main file for training and testing
    └── README.md              # This document
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/1BDH1C66BCKBe7ge8G-rBaj1j3p0iR0TC?usp=sharing), then place the downloaded contents under the correspond dataset folder such as `./data/SD`.

## Quick Start
1. Download datasets and place them under `./data`
2. We provide pre-trained weights of results in the paper and the detail configurations under the folder `./config`. 

---

To test PatchSTG w/o ST-TTC (i.e., Normal test) on different datasets, 
First, you should make sure that **line 391** in [main.py](/home/weichen/stg_project/ST-TTC/large_scale_scenario/main.py) file contains the following:
```
solver.test()
```
and then, you can execute the Python file in the terminal:
```
python main.py --config ./config/CA.conf
python main.py --config ./config/GBA.conf
python main.py --config ./config/GLA.conf
python main.py --config ./config/SD.conf
```

---

To test PatchSTG w/ ST-TTC on different datasets,
First, you should make sure that **line 391** in [main.py](/home/weichen/stg_project/ST-TTC/large_scale_scenario/main.py) file contains the following:
```
solver.test_with_ttc()
```
and then, you can execute the Python file in the terminal:
```
python main.py --config ./config/CA.conf
python main.py --config ./config/GBA.conf
python main.py --config ./config/GLA.conf
python main.py --config ./config/SD.conf
```

