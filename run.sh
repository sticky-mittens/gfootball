# even even more
nohup python -u learning_with_option_templates.py --type easy --gamma 0.93 --learn_freq 3 --lr 0.001 --mem_size 20 > output_longRun_easy_0.93_3_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type easy --gamma 0.99 --learn_freq 3 --lr 0.001 --mem_size 20 > output_longRun_easy_0.99_3_0.001_20_5_500.log &

# even more koa03

# nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_easy_0.9_1_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type easy --gamma 0.93 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_easy_0.93_1_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type easy --gamma 0.95 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_easy_0.95_1_0.001_20_5_500.log &
# nohup python -u learning_with_option_templates.py --type easy --gamma 0.97 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_easy_0.97_1_0.001_20_5_500.log &
# nohup python -u learning_with_option_templates.py --type easy --gamma 0.99 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_easy_0.99_1_0.001_20_5_500.log &

nohup python -u learning_with_option_templates.py --type medium --gamma 0.9 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_medium_0.9_1_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type medium --gamma 0.93 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_medium_0.93_1_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type medium --gamma 0.95 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_medium_0.95_1_0.001_20_5_500.log &
# nohup python -u learning_with_option_templates.py --type medium --gamma 0.97 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_medium_0.97_1_0.001_20_5_500.log &
nohup python -u learning_with_option_templates.py --type medium --gamma 0.99 --learn_freq 1 --lr 0.001 --mem_size 20 > output_longRun_medium_0.99_1_0.001_20_5_500.log &

nohup python -u learning_with_option_templates.py --type medium --gamma 0.93 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_longRun_medium_0.93_10_0.01_10_6_500.log &
# nohup python -u learning_with_option_templates.py --type medium --gamma 0.95 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_longRun_medium_0.95_10_0.01_10_6_500.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type medium --gamma 0.97 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_longRun_medium_0.97_10_0.01_10_6_500.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type medium --gamma 0.99 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_longRun_medium_0.99_10_0.01_10_6_500.log &

# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.96 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.96_10_0.01_10_7_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.97 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.97_10_0.01_10_7_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.98 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.98_10_0.01_10_7_500.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.96 --learn_freq 10 --lr 0.05 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.96_10_0.05_10_7_500.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.97 --learn_freq 10 --lr 0.05 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.97_10_0.05_10_7_500.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.98 --learn_freq 10 --lr 0.05 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.98_10_0.05_10_7_500.log &

CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.96 --learn_freq 10 --lr 0.005 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.96_10_0.005_10_7_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.97 --learn_freq 10 --lr 0.005 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.97_10_0.005_10_7_500.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.98 --learn_freq 10 --lr 0.005 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.98_10_0.005_10_7_500.log &


# ubuntu

# # 1:2 ratio (learn freq expt) for easy
# CUDA_VISIBLE_DEVICES=2 nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 10 --lr 0.001 --mem_size 20 > output_longRun_easy_0.9_10_0.001_20_5_500.log &
# CUDA_VISIBLE_DEVICES=2 nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 20 --lr 0.001 --mem_size 40 > output_longRun_easy_0.9_20_0.001_40_5_500.log &
# # 1:1 ratio (learn freq expt) for easy
# CUDA_VISIBLE_DEVICES=2 nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 10 --lr 0.001 --mem_size 10 > output_longRun_easy_0.9_10_0.001_10_5_500.log &
# CUDA_VISIBLE_DEVICES=2 nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 20 --lr 0.001 --mem_size 20 > output_longRun_easy_0.9_20_0.001_20_5_500.log &






# koa03

# # inc lr for easy
# nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 --learn_freq 1 --lr 0.01 --mem_size 20 > output_longRun_easy_0.9_1_0.01_20_5_500.log &
# nohup python -u learning_with_option_templates.py --type easy --gamma 0.99 --learn_freq 1 --lr 0.01 --mem_size 20 > output_longRun_easy_0.99_1_0.01_20_5_500.log &
# # inc lr for hard, gammas in between
# nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 1 --lr 0.01 --mem_size 20 > output_longRun_hard_0.95_1_0.01_20_5_500.log &
# nohup python -u learning_with_option_templates.py --type hard --gamma 0.97 --learn_freq 1 --lr 0.01 --mem_size 20 > output_longRun_hard_0.97_1_0.01_20_5_500.log &
# # inc lr for hard, gamma in between, bigger_net
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 1 --lr 0.01 --mem_size 20 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.95_1_0.01_20_7_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 1 --lr 0.01 --mem_size 20 --n_layers 7 --n_hidden_units 700 > output_longRun_hard_0.95_1_0.01_20_7_700.log &
# # 1:2 ratio, inc lr for hard, gamma in between, bigger_net
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 10 --lr 0.01 --mem_size 20 --n_layers 7 --n_hidden_units 500 > output_longRun_hard_0.95_10_0.01_20_7_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 20 --lr 0.01 --mem_size 40 --n_layers 7 --n_hidden_units 700 > output_longRun_hard_0.95_20_0.01_40_7_700.log &



# # 1:1 ratio (learn freq expt) for hard
# nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 10 --lr 0.001 --mem_size 10 > output_longRun_hard_0.95_10_0.001_10_5_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 20 --lr 0.001 --mem_size 20 > output_longRun_hard_0.95_20_0.001_20_5_500.log &
# # 1:1 ratio with higher lr for hard
# nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 10 --lr 0.01 --mem_size 10 > output_longRun_hard_0.95_10_0.01_10_5_500.log &
# CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 20 --lr 0.01 --mem_size 20 > output_longRun_hard_0.95_20_0.01_20_5_500.log &




# mac


# # 1:2 ratio (learn freq expt) for hard
# nohup python3 -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 10 --lr 0.001 --mem_size 20 > output_longRun_hard_0.95_10_0.001_20_5_500.log &
# nohup python3 -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 20 --lr 0.001 --mem_size 40 > output_longRun_hard_0.95_20_0.001_40_5_500.log &
# # 1:2 ratio with higher lr for hard
# nohup python3 -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 10 --lr 0.01 --mem_size 20 > output_longRun_hard_0.95_10_0.01_20_5_500.log &
# nohup python3 -u learning_with_option_templates.py --type hard --gamma 0.95 --learn_freq 20 --lr 0.01 --mem_size 40 > output_longRun_hard_0.95_20_0.01_40_5_500.log &










# # ubuntu laptop

# nohup python -u learning_with_option_templates.py --type easy --gamma 0.9 > output_longRun_easy_0.9.log &

# nohup python -u learning_with_option_templates.py --type easy --gamma 0.99 > output_longRun_easy_0.99.log &

# nohup python -u learning_with_option_templates.py --type easy --gamma 0.99 --use_target --target_update 1 > output_longRun_easy_0.99_useTarget_1.log &

# nohup python -u learning_with_option_templates.py --type hard --gamma 0.9 > output_longRun_hard_0.9.log &

# nohup python -u learning_with_option_templates.py --type hard --gamma 0.99 > output_longRun_hard_0.99.log &


# # mac docker

# nohup python3 -u learning_with_option_templates.py --type medium --gamma 0.9 > output_longRun_medium_0.9.log &

# nohup python3 -u learning_with_option_templates.py --type medium --gamma 0.99 > output_longRun_medium_0.99.log &


# # koa03

# nohup python3 -u learning_with_option_templates.py --type easy --gamma 0.99 --use_target --target_update 10 > output_longRun_easy_0.99_useTarget_10.log &

# nohup python3 -u learning_with_option_templates.py --type easy --gamma 0.99 --use_target --target_update 50 > output_longRun_easy_0.99_useTarget_50.log &