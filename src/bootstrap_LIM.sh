#!/bin/bash
for i in {39..1000}; do
	echo $i
	echo "CUDA_VISIBLE_DEVICES=3 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/LIM_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/LIM_bootstrapping/trial_$i/ >> bootstrap_LIM.txt"
	CUDA_VISIBLE_DEVICES=3 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/LIM_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/LIM_bootstrapping/trial_$i/ >> bootstrap_LIM.txt
done