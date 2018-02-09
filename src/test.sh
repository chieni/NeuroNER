#!/bin/bash
echo "CUDA_VISIBLE_DEVICES=$4 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/$1/$4 --pretrained_model_folder=../trained_models/$2 --output_folder=../output/$1/$4/ >>  $3"
echo $1
echo $2
echo $3
echo $4
echo $5
echo $6
#CUDA_VISIBLE_DEVICES=$4 python3.5 main.py --train_model=False --use_pretrained_model=True --dataset_text_folder=../data/$1/$5 --pretrained_model_folder=../trained_models/$2 --output_folder=../output/$1/$5/ >>  $3