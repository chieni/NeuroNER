#!/bin/bash
echo $1
echo "../data/over75_bootstrapping/trial_$1/"
fam_start=$(date)
CUDA_VISIBLE_DEVICES=1 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$1/ --pretrained_model_folder=../trained_models/fam_model --output_folder=../output/overxw75_bootstrapping/FAM/trial_$1 >> bootstrap_over75_FAM.txt
echo -e "FAM\t$fam_start\t$(date)" >> fam_runtime.txt