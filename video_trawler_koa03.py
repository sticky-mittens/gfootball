
import os
from glob import glob

for type, seed, ep_num in [("easy", 10, 16), ("medium", 10, 10), ("hard", 10, 46)]:
    folder = f"videos_while_solving_{type}_seed{seed}"
    all_avis = glob(f"{folder}/*.avi")

    print(f'type {type}, seed {seed}, ep in log {ep_num}, {all_avis[ep_num+1]}')
