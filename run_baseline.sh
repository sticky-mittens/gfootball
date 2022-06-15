

# baseline

CUDA_VISIBLE_DEVICES=1 nohup python -u learning_baseline.py --type easy --gamma 0.9 > output_baseline_easy_0.9.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_baseline.py --type easy --gamma 0.9 --learn_freq 10 --lr 0.001 --mem_size 20 > output_baseline_easy_0.9_10_0.001_20_5_500.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u learning_baseline.py --type medium --gamma 0.93 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_baseline_medium_0.93_10_0.01_10_6_500.log &
python -u learning_baseline.py --type medium --gamma 0.97 --learn_freq 1 --lr 0.001 --mem_size 20 > output_baseline_medium_0.97_1_0.001_20_5_500.log &
python -u learning_baseline.py --type medium --gamma 0.99 --learn_freq 1 --lr 0.001 --mem_size 20 > output_baseline_medium_0.99_1_0.001_20_5_500.log &

# nohup python -u learning_baseline.py --type hard --gamma 0.96 --learn_freq 10 --lr 0.05 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_baseline_hard_0.96_10_0.05_10_7_500.log &
nohup python -u learning_baseline.py --type hard --gamma 0.96 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_baseline_hard_0.96_10_0.01_10_7_500.log &
