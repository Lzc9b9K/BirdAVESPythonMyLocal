import os
# import shutil
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import soundfile as sf
from datetime import timedelta
from sklearn.model_selection import train_test_split
# from scipy.io import wavfile
from scipy.stats import mode
import h5py
import numpy as np

SAMPLE_RATE = 16000
# DURATION = 16 * 60 # 16 mins

# SEGMENT_DURATION_S = 16000 # 1s samples
# SEGMENT_DURATION_GROUP_NUMS = 49 # 1s embeddings
# SAMPLES_PER_GROUP = 326 # 49 * 326 = 15974 - 16000 =  -26

# TRAIN_SAMPLES_LENGTH = 15360000 # 16000 * 60 * 16
# TRAIN_EMBEDDINGS_LENGTH = 47999
# TRAIN_SAMPLES_PER_GROUP = 320

# VAL_SAMPLES_LENGTH = 3840000 # 16000 * 60 * 4
# VAL_EMBEDDINGS_LENGTH = 11999
# VAL_SAMPLES_PER_GROUP = 320

TRAIN_SAMPLES_LENGTH = 19200000 # 16000 * 60 * 20
TRAIN_EMBEDDINGS_LENGTH = 47999
TRAIN_SAMPLES_PER_GROUP = 320


CALL_TYPES = [
    "foodCall", # Food call
    # "pleasureCall", #     Pleasuse call
    # "distressCall", # Distress call
    # "FoPl", # Food call + Pleasure call 
    # "FoDi", # Food call + Distress call
    # "Fe", # Fear trill
    "other", # Other
]

def get_call_type(row):
    # if row['foodCall'] >= 1 and row['pleasureCall'] >= 1:
    #     # return "FoPl"
    #     return "Other"
    # elif row['foodCall'] >= 1 and row['distressCall'] >= 1:
    #     # return "FoDi"
    #     return "Other"
    # if row['foodCall'] >= 1:
    #     return "foodCall"
    # # elif row['distressCall'] >= 1:
    # #     return "other"
    # # elif row['pleasureCall'] >= 1:
    # #     return "other"
    # # elif row['fearTrill'] >= 1:
    # #     return "other"
    # elif row['other'] >= 1:
    #     return "other"
    # else:
    #     return None
    # print(f"row : {row}")
    # print(f"np.array(row, dtype=np.int32) : {np.array(row, dtype=np.int32)}")
    if np.array_equal(np.array(row, dtype=np.int32), np.array([1,0], dtype=np.int32)):
        return "foodCall"
    elif np.array_equal(np.array(row, dtype=np.int32), np.array([0,1], dtype=np.int32)):
        return "other"
    else:
        # return None
        return "other"

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"

def create_dataset(embedding_file, label_file, output_dir):
    embedding_file_name = "trainSignalPure.h5"
    label_file_name = "trainSignal_compressed_label.h5"
    config_file_name = "full_dataset_config.csv"

    # embedding_file_name = "bandPassed_valSignal.h5"
    # label_file_name = "valSignal_compressed__label.h5"
    # config_file_name = "val_dataset_config.csv"

    full_dataset_info = []
    call_type_counts = {call_type: 0 for call_type in CALL_TYPES}

    # audio = AudioSegment.from_wav(audio_file)
    # sr, audio = wavfile.read(audio_file)
    # embeddings_df = pd.read_hdf(embedding_file, key="tensor_dataset")
    # embeddings_df = pd.read_hdf(embedding_file, "tensor_dataset")
    embeddings_file = h5py.File(embedding_file) 
    embeddings_dataset = embeddings_file["./tensor_dataset"]
    # print(embeddings_dataset)
    
    # arrary_embedding_df = embeddings_df.to_numpy()
    labels_df = pd.read_csv(label_file)

    labels_df['other'] = labels_df['other'] | labels_df['fearTrill'] | labels_df['distressCall'] | labels_df['pleasureCall']
    labels_df.drop(columns=['fearTrill'], inplace=True)
    labels_df.drop(columns=['distressCall'], inplace=True)
    labels_df.drop(columns=['pleasureCall'], inplace=True)
    one_hot_count = labels_df.sum(axis=1)
    # labels_df.loc[one_hot_count != 1, ['foodCall', 'pleasureCall', 'distressCall']] = 0
    labels_df.loc[one_hot_count > 1, 'foodcall'] = 0
    labels_df.loc[one_hot_count > 1, 'other'] = 1
    # labels_df = labels_df[['foodCall', 'pleasureCall', 'distressCall', 'other']]
    labels_df = labels_df[['foodCall', 'other']]
    
    start_sample = 0
    start_embedding = 0
    total_compressed_labels = []
    while start_embedding < len(embeddings_dataset):
        end_embedding = start_embedding + 1
        
        start_sample = start_embedding * TRAIN_SAMPLES_PER_GROUP + 1
        end_sample = end_embedding * TRAIN_SAMPLES_PER_GROUP

        # start_sample = start_embedding * VAL_SAMPLES_PER_GROUP + 1
        # end_sample = end_embedding * VAL_SAMPLES_PER_GROUP

        

        # check whole label
        # labels = labels_df.iloc[start_sample:end_sample].sum()

        
        # compress label
        group_labels = labels_df.iloc[start_sample : end_sample]
        arrary_group_labels = group_labels.to_numpy(dtype=np.int32)
        group_mode = mode(arrary_group_labels, axis=0, keepdims=True).mode[0]
        # total_compressed_labels.append(group_mode)

        # print(f"group mode : {group_mode}")
        call_type = get_call_type(group_mode)
        if (call_type == "other"):
            total_compressed_labels.append([0,1])
        else:
            total_compressed_labels.append(group_mode)


        # if call_type:
        # signal
        timestamp = format_timestamp(start_sample / SAMPLE_RATE)
        # print("start_sample", start_sample)
        print(f"timestamp: {timestamp}, call_type: {call_type}")
        # print(f"group mode : {group_mode}")
        full_dataset_info.append({
            "call_type": call_type, 
            "embedding_index": start_embedding,
            "embedding_path": os.path.join(output_dir, embedding_file_name),                
            "label_path": os.path.join(output_dir, label_file_name),
            "subset": "train"
            })
        call_type_counts[call_type] += 1

        start_embedding = end_embedding
    
    # save labels
    # print(f"total_compressed_labels, {total_compressed_labels}")
    # compressed_label_df = pd.DataFrame(total_compressed_labels, columns=['foodCall', 'other'])
    # compressed_label_df.to_csv(os.path.join(output_dir, label_file_name), index=False)

    # compressed_label_df.to_hdf(os.path.join(output_dir, label_file_name), key="tensor_dataset", mode="w")
    with h5py.File(os.path.join(output_dir, label_file_name), "w") as hdf:
        hdf.create_dataset("tensor_dataset", data=total_compressed_labels)

    # dataset_df = pd.DataFrame(dataset_info, columns=['filepath', 'call_type'])
    full_dataset_df = pd.DataFrame(full_dataset_info)
    full_dataset_df.to_csv(os.path.join(output_dir, config_file_name), index=False)

    print("Dataset size:", len(full_dataset_info))
    # print("Call types:", dataset_df['call_type'].unique())
    for call_type, count in call_type_counts.items():
        print(f"{call_type}: {count}")

def save_train_and_test_dataset(fulldataset_file, output_dir):
    train_info = []
    test_info = []

    fulldataset = pd.read_csv(fulldataset_file)

    for call_type in CALL_TYPES:
        # call_type_data = [info for info in fulldataset.columns if info['call_type'] == call_type]
        call_type_data = fulldataset[fulldataset['call_type'] == call_type]
        # print(call_type_data)
        train_data, test_data = train_test_split(call_type_data, train_size=0.8, test_size=0.2)
        # print(train_data)

        # train_info.extend(train_data.to_dict(orient='records'))
        # test_info.extend(test_data.to_dict(orient='records'))
        for _, row in train_data.iterrows():
            info = row.to_dict()
            info['subset'] = 'train'
            # info['filepath'] = os.path.join(train_output_dir, info['filename'])
            # info['labelpath'] = os.path.join(train_output_dir, f"{info['filename'][:-4]}.csv")
            train_info.append(info)
        
        for _, row in test_data.iterrows():
            info = row.to_dict()
            info['subset'] = 'val'
            # info['filepath'] = os.path.join(test_output_dir, info['filename'])
            # info['labelpath'] = os.path.join(test_output_dir, f"{info['filename'][:-4]}.csv")
            test_info.append(info)


    train_df = pd.DataFrame(train_info)
    train_df.to_csv(os.path.join(output_dir, 'train_dataset_config_two.csv'), index=False)
    test_df = pd.DataFrame(test_info)
    test_df.to_csv(os.path.join(output_dir, 'val_dataset_config_two.csv'), index=False)

if __name__ == "__main__":
    save_dir = "./dataset/processed_embedding_label_dataset_20241203_two"

    # create full dataset config
    origin_features_file = f"{save_dir}/trainSignalPure.h5"
    # origin_features_file = f"{save_dir}/bandPassed_trainSignal.h5"
    origin_label_file = f"{save_dir}/trainSignal_label.csv"
    create_dataset(
        origin_features_file, 
        origin_label_file, 
        save_dir
    )

    # create train & val dataset config
    # total_dataset_config = "./dataset/nonprocessed_embedding_label_dataset_20241122/full_dataset_config.csv"
    # save_train_and_test_dataset(total_dataset_config, save_dir)

    # origin_features_file = "./dataset/embedding_label_dataset_20241121/bandPassed_valSignal.h5"
    # origin_label_file = "./dataset/embedding_label_dataset_20241121/valSignal_label.csv"

    # create_dataset(
    #     origin_features_file, 
    #     origin_label_file, 
    #     save_dir
    # )

    
