#!/bin/bash

for i in {1..10}; do
	echo $i
	start_time=$(date)
	echo "$start_time"
	# Write line to bootstrap
	CUDA_VISIBLE_DEVICES=2 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over75_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/lim_model --output_folder=../output/time_test/LIM/trial_$i
	echo -e "LIM\t$start_time\t$(date)" >> test_runtime.txt
done