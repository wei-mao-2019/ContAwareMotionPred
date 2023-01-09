#!/bin/bash

python exp_gta_stage1_cont.py --cfg gta_stage1_PVCNN2_DCT_CONT --is_amp --is_dist --gpus 2
python exp_gta_stage2_pose.py --cfg gta_stage2_GRU_POSE --is_amp --is_dist --gpus 2

python eval_gta_stats.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT --cfg gta_stage2_GRU_POSE --iter 1 --iter_cont 1
python eval_gta_stats.py --cfg_cont gta_stage1_PVCNN2_DCT_CONT --cfg gta_stage2_GRU_POSE --iter 1 --iter_cont 1 --w_est_cont
