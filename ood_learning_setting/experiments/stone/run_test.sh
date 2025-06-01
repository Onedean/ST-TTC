## Spatial Temporal OOD
# SD 2019

# 10 %: seed 0 2 4

python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 0 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1 --mode test &
python experiments/stone/main.py --device cuda:2 --dataset SD --years 2019 --model_name stone --seed 2 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1 --mode test &
python experiments/stone/main.py --device cuda:3 --dataset SD --years 2019 --model_name stone --seed 4 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.1 --mode test &

# 15 %: seed 5 6 7

python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 5 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.15 --mode test &
python experiments/stone/main.py --device cuda:2 --dataset SD --years 2019 --model_name stone --seed 6 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.15 --mode test &
python experiments/stone/main.py --device cuda:3 --dataset SD --years 2019 --model_name stone --seed 7 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.15 --mode test &

# 20 %: seed 8 9 10

python experiments/stone/main.py --device cuda:1 --dataset SD --years 2019 --model_name stone --seed 8 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.20 --mode test &
python experiments/stone/main.py --device cuda:2 --dataset SD --years 2019 --model_name stone --seed 9 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.20 --mode test &
python experiments/stone/main.py --device cuda:3 --dataset SD --years 2019 --model_name stone --seed 10 --bs 1 --c 10 --ood 1 --tood 1 --new_node_ratio 0.20 --mode test &

