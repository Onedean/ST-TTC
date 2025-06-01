#!/bin/bash


python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 42

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 43

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 44

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 45

python stkec_main.py --conf conf/AIR/stkec.json --gpuid 1 --seed 46


python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 42

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 43

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 44

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 45

python main.py --conf conf/AIR/eac.json --gpuid 0 --seed 46
