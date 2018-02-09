#!/bin/bash
# Training LIM model on CAR data
for i in {1..1001}; do
	echo $i
	echo 'LIM'
	CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/CIM_bootstrapping/LIM/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/CIM_bootstrapping/LIM/trial_$i/ >> bootstrap_CIM_LIM.txt
done