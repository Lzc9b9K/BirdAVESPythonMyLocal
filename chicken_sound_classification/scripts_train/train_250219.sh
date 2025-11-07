#!/bin/bash
root_data_dir="D:/meng/01_Code/aves/chicken_sound_classification/dataset"
root_save_dir="D:/meng/01_Code/aves/chicken_sound_classification/2025_05_trained_model"
gpu_id=0

echo "Training shell script is starting"

CUDA_VISIBLE_DEVICES="$gpu_id" \
C:/Users/miniP/miniconda3/envs/aves/python.exe \
D:/meng/01_Code/aves/chicken_sound_classification/train_network.py \
    --logger_project svm_3_layer_model_0220 \
    --logger_name bandPassed_dataset_2 \
    --logger_save_dir "$root_save_dir" \
    --cpkt_save_every_n_epochs 5 \
    --accelerator gpu \
    --devices 1 \
    --max_epochs 100 \
    --classes_nums 2 \
    --batch_size 4096 \
    --val_batch_size 2 \
    --num_workers 1 \
    --learning_rate 1e-3 \
    --embedding_dim 1024 \
    --sample_rate 16000 \
    --train_dataset_config "$root_data_dir/train_data/bandPassedDatas/bandPassed_full_dataset_config.csv" \
    --val_dataset_config "$root_data_dir/val_data/bandPassed_val_dataset_config.csv" \
    --test_dataset_config "$root_data_dir/val_data/bandPassed_val_dataset_config.csv" \

read -n 1 -s -r -p "Press any key to continue..."
