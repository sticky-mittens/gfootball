from distutils import file_util
from cv2 import findChessboardCornersSBWithMeta
import matplotlib.pyplot as plt 
import numpy as np
import os
complete = []
for file in [
                "full_run_logs/output_lvl2_seed20_easy_0.93_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed20_medium_0.93_10_0.01_10_6_500.log", "full_run_logs/output_lvl2_seed20_hard_0.96_10_0.005_10_7_500.log",
                "full_run_logs/output_lvl2_seed2_easy_0.93_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed2_easy_0.99_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed2_medium_0.93_10_0.01_10_6_500.log", "full_run_logs/output_lvl2_seed2_hard_0.96_10_0.005_10_7_500.log",
                "full_run_logs/output_lvl2_seed5_easy_0.93_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed5_easy_0.99_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed5_medium_0.93_10_0.01_10_6_500.log", "full_run_logs/output_lvl2_seed5_hard_0.96_10_0.005_10_7_500.log",
                "full_run_logs/output_lvl2_seed10_easy_0.93_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed10_easy_0.99_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed10_medium_0.93_10_0.01_10_6_500.log", "full_run_logs/output_lvl2_seed10_hard_0.96_10_0.005_10_7_500.log",
                "full_run_logs/output_lvl2_seed3_easy_0.93_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed3_easy_0.99_1_0.001_20_5_500.log", "full_run_logs/output_lvl2_seed3_medium_0.93_10_0.01_10_6_500.log", "full_run_logs/output_lvl2_seed3_hard_0.96_10_0.005_10_7_500.log",
                "./output_baseline_easy_0.9.log", "./output_baseline_easy_0.9_10_0.001_20_5_500.log", "./output_baseline_medium_0.93_10_0.01_10_6_500.log", "./output_baseline_medium_0.97_1_0.001_20_5_500.log", "./output_baseline_medium_0.99_1_0.001_20_5_500.log", "./output_baseline_hard_0.96_10_0.01_10_7_500.log",
                "logs/output_longRun_easy_0.9.log", "logs/output_longRun_easy_0.99.log", # from ubuntu
                #"logs/output_baseline_easy_0.9.log", "logs/output_baseline_easy_0.9_10_0.001_20_5_500.log", "logs/output_baseline_medium_0.93_10_0.01_10_6_500.log", "logs/output_baseline_medium_0.97_1_0.001_20_5_500.log", "logs/output_baseline_medium_0.99_1_0.001_20_5_500.log", "logs/output_baseline_hard_0.96_10_0.05_10_7_500.log", "logs/output_baseline_hard_0.96_10_0.01_10_7_500.log", 
                "logs/output_longRun_easy_0.93_3_0.001_20_5_500.log", "logs/output_longRun_easy_0.99_3_0.001_20_5_500.log",
                "logs/output_longRun_easy_0.93_1_0.001_20_5_500.log", "logs/output_longRun_easy_0.95_1_0.001_20_5_500.log", "logs/output_longRun_easy_0.97_1_0.001_20_5_500.log", "logs/output_longRun_easy_0.99_1_0.001_20_5_500.log", 
                "logs/output_longRun_medium_0.9_1_0.001_20_5_500.log", "logs/output_longRun_medium_0.93_1_0.001_20_5_500.log", "logs/output_longRun_medium_0.95_1_0.001_20_5_500.log", "logs/output_longRun_medium_0.97_1_0.001_20_5_500.log", "logs/output_longRun_medium_0.99_1_0.001_20_5_500.log", 
                "logs/output_longRun_medium_0.93_10_0.01_10_6_500.log", "logs/output_longRun_medium_0.95_10_0.01_10_6_500.log", "logs/output_longRun_medium_0.97_10_0.01_10_6_500.log", "logs/output_longRun_medium_0.99_10_0.01_10_6_500.log",
                "logs/output_longRun_hard_0.96_10_0.01_10_7_500.log", "logs/output_longRun_hard_0.97_10_0.01_10_7_500.log", "logs/output_longRun_hard_0.98_10_0.01_10_7_500.log", "logs/output_longRun_hard_0.96_10_0.05_10_7_500.log", "logs/output_longRun_hard_0.97_10_0.05_10_7_500.log", "logs/output_longRun_hard_0.98_10_0.05_10_7_500.log", "logs/output_longRun_hard_0.96_10_0.005_10_7_500.log", "logs/output_longRun_hard_0.97_10_0.005_10_7_500.log", "logs/output_longRun_hard_0.98_10_0.005_10_7_500.log", 
                ]:
# for file in ["logs/output_longRun_easy_0.9_1_0.01_20_5_500.log", "logs/output_longRun_easy_0.99_1_0.01_20_5_500.log", "logs/output_longRun_hard_0.95_1_0.01_20_5_500.log", "logs/output_longRun_hard_0.97_1_0.01_20_5_500.log", "logs/output_longRun_hard_0.95_1_0.01_20_7_500.log", "logs/output_longRun_hard_0.95_1_0.01_20_7_700.log", "logs/output_longRun_hard_0.95_10_0.01_20_7_500.log", "logs/output_longRun_hard_0.95_20_0.01_40_7_700.log", "logs/output_longRun_hard_0.95_10_0.001_10_5_500.log", "logs/output_longRun_hard_0.95_20_0.001_20_5_500.log", "logs/output_longRun_hard_0.95_10_0.01_10_5_500.log", "logs/output_longRun_hard_0.95_20_0.01_20_5_500.log"]:
    if file in complete:
        print(f'{file} is repeated.')
    else:
        complete.append(file)
    
    rewards = []
    score_diffs = []
    exploring = True
    with open(file, "r") as f:
        for line in f.readlines():
            if 'reward' in line and 'right preds' not in line:
                if 'eps 0.001' in line and exploring:
                    episode_info = line.split(',')[0]
                    exploration_died_at_ep = float(episode_info.split(' ')[-1])
                    exploring = False

                reward_info = line.split(',')[3]
                reward = float(reward_info.split(' ')[-1])

                score_diff_info = line.split(',')[4]
                score_diff = float(score_diff_info.split(' ')[-1])

                rewards.append(reward)
                score_diffs.append(score_diff)

    # if 'easy_0.9.log' in file:
    #     rewards = rewards[:-20]
    # else:
    #     rewards= rewards

    N = 5
    avg_rewards = []
    avg_score_diffs = []
    unsorted_tuples = []
    
    all_score_diffs = []
    episodes = []
    best_avg = -200
    for i in range(0, len(rewards)):#, N):
        if i >= exploration_died_at_ep:
            episodes.append(i)

            end = i + N if (i+N <= len(rewards)) else len(rewards)
            N_rewards = rewards[i:end]
            N_avg_reward = sum(N_rewards)/N

            N_score_diffs = score_diffs[i:end]
            N_avg_score_diff = sum(N_score_diffs)/N

            avg_rewards.append(N_avg_reward)
            avg_score_diffs.append(N_avg_score_diff)
            if 'baseline' not in file:
                if i <=650:
                    unsorted_tuples.append((N_avg_score_diff, i, round(np.std(N_score_diffs), 3)))
            else:
                if i <=1250:
                    unsorted_tuples.append((N_avg_score_diff, i, round(np.std(N_score_diffs), 3)))

            all_score_diffs.extend(N_score_diffs)
            
            # if N_avg_score_diff >= best_avg:
            #     best_avg = N_avg_score_diff
            #     corr_std = np.std(N_score_diffs)
            #     best_ep = i
            #     best_end = end

    plt.figure()
    plt.plot(episodes, avg_rewards)
    plt.plot(episodes, avg_score_diffs)
    plt.legend(['moving_avg_reward', f'moving_avg_score_diff'])
    print(file)

    sorted_tuples = sorted(unsorted_tuples, key = lambda x: x[0])
    for idx, (best_avg, best_ep, corr_std) in enumerate(reversed(sorted_tuples)):
        if idx >= 10:
            break
        best_end = best_ep + N
        best_info = f'best            {best_avg} @ ep {best_ep}:{best_end} = steps {best_ep*3000}:{best_end*3000}'
        print(best_info)
    avg_of_avgs_info = f'avg of all      {np.average(all_score_diffs)} +- {np.std(all_score_diffs)}'
    print(avg_of_avgs_info)
    
    # for ep_chosen in [430, 355, 335, 160, ]:
    #     N2 = 5
    #     best_avg = -200
    #     for e in range(ep_chosen-5, ep_chosen+5):
    #         few_score_diffs = score_diffs[e:e+N2]
    #         few_avg = np.average(few_score_diffs)
    #         if few_avg >= best_avg:
    #             best_avg = few_avg
    #             corr_std = np.std(few_score_diffs)
    #             best_ep = e
    #             best_end = e+N2
    #     best_chosen_info = f'best around {ep_chosen} is {best_avg} +- {corr_std} @ ep {best_ep}:{best_end} = steps {best_ep*3000}:{best_end*3000}'
    #     print(best_chosen_info)
    print('\n')

    if 'full_run_logs' in file:
        suffix = file[14:-4]
    elif 'logs' in file:
        suffix = file[5:-4]
    else:
        suffix = file[2:-4]
    plt.title(file)
    os.makedirs('./result_plots', exist_ok=True)
    plt.savefig(f'./result_plots/{suffix}.png')
    # plt.show()

    
second_time_running_this_file = True
if second_time_running_this_file:
    data_from_first_time = {
        'easy_ours': [  4.8, # or 5.4
                        6.4,
                        5.6,
                        6.0,
                        7.2,
                        6.2,
                        4.0,
                        6.0 ], 
        'medium_ours': [2.0, 3.4, 3.6],
        'hard_ours': [2.2, 1.8, 2.0, 1.8, 2.2],
        'easy_base': [1.6, 1.2, 0.8],
        'medium_base': [1.2, 0.8, 0.4],
        'hard_base': [0.4, 0.2, 0.18],
    }
    for k, v in data_from_first_time.items():
        print(k, round(np.average(v), 2), ' +- ', round(np.std(v), 2))