#!/bin/bash
# Training LIM model on CAR data
for i in {0..1012}; do
	echo $i
	echo "CUDA_VISIBLE_DEVICES=3 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/CAR_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/CIM_bootstrapping/trial_$i/ >> bootstrap_CIM.txt"
	CUDA_VISIBLE_DEVICES=3 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/CAR_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/CIM_bootstrapping/trial_$i/ >> bootstrap_CIM.txt
done