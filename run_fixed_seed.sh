# Ours but with new pre conditions for charge_goal and just_shoot









# d=1
# s=2001 # 2, 10, 23, 2001
# easy_gamma=0.9 # 0.99, 0.9

# CUDA_VISIBLE_DEVICES=${d} nohup python -u learning_with_option_templates.py --seed ${s} --type easy   --gamma ${easy_gamma} --learn_freq 1  --lr 0.001 --mem_size 20 --n_layers 5 --n_hidden_units 500 > output_seed${s}_l2_easy_${easy_gamma}_1_0.001_20_5_500.log &

# # CUDA_VISIBLE_DEVICES=${d} nohup python -u learning_with_option_templates.py --seed ${s} --type medium --gamma 0.93 --learn_freq 10 --lr 0.01  --mem_size 10 --n_layers 6 --n_hidden_units 500 > output_seed${s}_l2_easy_0.93_10_0.01_10_6_500.log &

# # CUDA_VISIBLE_DEVICES=${d} nohup python -u learning_with_option_templates.py --seed ${s} --type hard   --gamma 0.96 --learn_freq 10 --lr 0.005 --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_seed${s}_l2_easy_0.96_10_0.005_10_7_500.log &

# # CUDA_VISIBLE_DEVICES=${d} nohup python -u learning_with_option_templates.py --seed ${s} --type hard   --gamma 0.96 --learn_freq 10 --lr 0.05  --mem_size 10 --n_layers 7 --n_hidden_units 500 > output_seed${s}_l2_easy_0.96_10_0.05_10_7_500.log &

# # Baseline



