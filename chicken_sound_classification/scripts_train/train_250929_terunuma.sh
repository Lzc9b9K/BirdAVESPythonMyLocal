#!/bin/bash
root_data_dir="/app/chicken_sound_classification/dataset"
root_save_dir="/app/chicken_sound_classification/250929_terunuma_trained_model"
gpu_id=0
echo "Training shell script is starting"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python3 /app/chicken_sound_classification/train_network.py \
    --logger_project fc_3_layer_model_250923_terunuma \
    --logger_name bandPassed_dataset \
    --logger_save_dir "$root_save_dir" \
    --cpkt_save_every_n_epochs 50 \
    --accelerator gpu \
    --devices "-1" \
    --max_epochs 100 \
    --classes_nums 2 \
    --train_batch_size 1024 \
    --val_batch_size 24 \
    --train_num_workers 23 \
    --val_num_workers 23 \
    --learning_rate 1e-3 \
    --embedding_dim 1024 \
    --sample_rate 16000 \
    --train_dataset_config "$root_data_dir/train_data/bandPassedDatas/bandPassed_full_dataset_config.csv" \
    --val_dataset_config "$root_data_dir/val_data/bandPassed_val_dataset_config.csv" \
    --test_dataset_config "$root_data_dir/val_data/bandPassed_val_dataset_config.csv" 
read -n 1 -s -r -p "Press any key to continue..."
