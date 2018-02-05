#!/bin/bash
for i in {1..1000}; do
	echo $i
	CUDA_VISIBLE_DEVICES=3 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/CAR_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/car_model output_folder_name >> bootstrap_CAR.txt
done