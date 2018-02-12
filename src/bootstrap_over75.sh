#!/bin/bash
python3.5 bootstrap_over75.py $1

# Run model for LIM, CAR, FAM
echo $1
start_time=$(date)
# Write line to bootstrap
CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/over75_bootstrapping/LIM/trial_$1 >> bootstrap_over75_LIM.txt
echo -e "LIM\t$start_time\t$(date)" >> runtime.txt

car_start=$(date)
CUDA_VISIBLE_DEVICES=0 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/car_model --output_folder=../output/over75_bootstrapping/CAR/trial_$1 >> bootstrap_over75_CAR.txt
echo -e "CAR\t$car_start\t$(date)" >> runtime.txt

fam_start=$(date)
CUDA_VISIBLE_DEVICES=1 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/fam_model --output_folder=../output/overxw75_bootstrapping/FAM/trial_$1 >> bootstrap_over75_FAM.txt
echo -e "FAM\t$fam_start\t$(date)" >> runtime.txt
# Remove the file folder
rm -r ../data/over_75_bootstrapping/trial_$