#!/bin/bash
#python3.5 bootstrap_over75.py $1

# Run model for LIM, CAR, FAM
echo $1
echo "../data/over75_bootstrapping/trial_$1/"
start_time=$(date)
# Write line to bootstrap
CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/over75_bootstrapping/LIM/trial_$1 >> bootstrap_over75_LIM.txt
echo -e "LIM\t$start_time\t$(date)" >> runtime.txt
# Remove the file folder
#rm -r ../data/over_75_bootstrapping/trial_$