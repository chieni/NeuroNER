#!/bin/bash

# $1 filenum
# Sample deploy
python3.5 bootstrap_over75.py $1

# Run model for LIM, CAR, FAM
echo $1
echo "CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/over_75_bootstrapping/LIM/trial_$1 >> bootstrap_over75_LIM.txt"
CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/over75_bootstrapping/LIM/trial_$1 >> bootstrap_over75_LIM.txt
CUDA_VISIBLE_DEVICES=0 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/car_model --output_folder=../output/over75_bootstrapping/CAR/trial_$1 >> bootstrap_over75_CAR.txt
CUDA_VISIBLE_DEVICES=1 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/fam_model --output_folder=../output/overxw75_bootstrapping/FAM/trial_$1 >> bootstrap_over75_FAM.txt
# Remove the file folder
rm -r ../data/over_75_bootstrapping/trial_$