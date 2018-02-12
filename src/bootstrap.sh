#!/bin/bash
echo $1
echo "../data/over75_bootstrapping/trial_$1/"
car_start=$(date)
CUDA_VISIBLE_DEVICES=0 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/car_model --output_folder=../output/over75_bootstrapping/CAR/trial_$1 >> bootstrap_over75_CAR.txt
echo -e "CAR\t$car_start\t$(date)" >> car_runtime.txt