for n in 12 13 # 0 1 2 3
do
    mkdir videos${n}
    mkdir videos${n}/vid_logs

    for s in 20 # 2 or 20
    do
        # CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode easy --v ${n} > videos${n}/vid_logs/easy_${s}.log &
        CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode medium --v ${n} > videos${n}/vid_logs/medium_${s}.log &
        # CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode hard --v ${n} > videos${n}/vid_logs/hard_${s}.log &
    done
done

# mkdir videos
# mkdir videos/vid_logs
# for s in 2 3 5 10
# do
#     CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode easy > videos/vid_logs/easy_${s}.log &
#     CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode medium > videos/vid_logs/medium_${s}.log &
#     CUDA_VISIBLE_DEVICES=1 nohup python -u play_composed.py --seed ${s} --mode hard > videos/vid_logs/hard_${s}.log &
# done