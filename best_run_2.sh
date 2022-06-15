for s in 2 3 5 10 # 20
do
    # CUDA_VISIBLE_DEVICES=0 nohup python -u learning_with_option_templates.py --seed ${s} --level 2 --type easy   --gamma 0.99 --learn_freq 1 --lr 0.001 --mem_size 20 --n_layers 5 --n_hidden_units 500 > full_run_logs/output_lvl2_seed${s}_easy_0.99_1_0.001_20_5_500.log &
    CUDA_VISIBLE_DEVICES=0 nohup python -u learning_with_option_templates.py --seed ${s} --level 2 --type easy   --gamma 0.93 --learn_freq 1 --lr 0.001 --mem_size 20 --n_layers 5 --n_hidden_units 500 > full_run_logs/output_lvl2_seed${s}_easy_0.93_1_0.001_20_5_500.log &

    CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --seed ${s} --level 2 --type medium --gamma 0.93 --learn_freq 10 --lr 0.01 --mem_size 10 --n_layers 6 --n_hidden_units 500 > full_run_logs/output_lvl2_seed${s}_medium_0.93_10_0.01_10_6_500.log &

    CUDA_VISIBLE_DEVICES=1 nohup python -u learning_with_option_templates.py --seed ${s} --level 2 --type hard   --gamma 0.96 --learn_freq 10 --lr 0.005 --mem_size 10 --n_layers 7 --n_hidden_units 500 > full_run_logs/output_lvl2_seed${s}_hard_0.96_10_0.005_10_7_500.log &
done