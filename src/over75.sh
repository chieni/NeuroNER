#!/bin/bash
echo $1
for i in {1..10}; do
	echo $i
	start_time=$(date)
	echo "$start_time"
	# Write line to bootstrap
	CUDA_VISIBLE_DEVICES=$1 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/over_75 --pretrained_model_folder=../trained_models/cod_model --output_folder=../output/time_test/COD/trial_$i
	echo -e "LIM\t$start_time\t$(date)" >> cod_test_runtime.txt
done