#!/bin/bash


python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 42

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 43

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 44

python stkec_main.py --conf conf/ENERGY-Wind/stkec.json --gpuid 1 --seed 45






python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 42

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 43 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.88.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 44 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.8834.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 45 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.8834.pkl"

python main.py --conf conf/ENERGY-Wind/eac.json --gpuid 0 --seed 46 --load_first_year 1 --first_year_model_path "log/ENERGY-Wind/eac-42/0/2.88.pkl" 
