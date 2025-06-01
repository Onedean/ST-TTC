#!/bin/bash



python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 47

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 48

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 49

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 50

python stkec_main.py --conf conf/PEMS/stkec.json --gpuid 0 --seed 51



python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 51

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 52

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 53

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 54

python main.py --conf conf/PEMS/eac.json --gpuid 1 --seed 55
