#!/bin/bash
for i in {1210..1274}; do
	echo $i
	echo "CUDA_VISIBLE_DEVICES=0 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/FAM_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/fam_model --output_folder=../output/FAM_bootstrapping/trial_$i/ >> bootstrap_FAM.txt"
	CUDA_VISIBLE_DEVICES=0 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/FAM_bootstrapping/trial_$i/ --pretrained_model_folder=../trained_models/fam_model --output_folder=../output/FAM_bootstrapping/trial_$i/ >> bootstrap_FAM.txt
done