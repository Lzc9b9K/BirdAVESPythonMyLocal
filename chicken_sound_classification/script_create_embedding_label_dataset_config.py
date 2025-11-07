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
# TRAIN_EMBEDDINGS_LENGTH = 59999
# VAL_SAMPLES_PER_GROUP = 320
# TRAIN_SAMPLES_PER_GROUP = 128
TRAIN_SAMPLES_PER_GROUP = 320


FOUR_CALL_TYPES = [
    "foodCall", # Food call
    "pleasureCall", #     Pleasuse call
    "distressCall", # Distress call
    # "FoPl", # Food call + Pleasure call 
    # "FoDi", # Food call + Distress call
    # "Fe", # Fear trill
    "other", # Other
]

TWO_CALL_TYPES = [
    "foodCall",
    "other",
]

def get_call_type(row):
    if (np.array_equal(np.array(row, dtype=np.int32), np.array([1,0,0,0], dtype=np.int32)) 
        or np.array_equal(np.array(row, dtype=np.int32), np.array([1,0], dtype=np.int32))):
        return "foodCall"
    elif np.array_equal(np.array(row, dtype=np.int32), np.array([0,1,0,0], dtype=np.int32)):
        return "pleasureCall"
    elif np.array_equal(np.array(row, dtype=np.int32), np.array([0,0,1,0], dtype=np.int32)):
        return "distressCall"
    else:
        return "other"

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"

def create_train_dataset(embedding_file, label_file, output_dir, n_classes=4, config_file_name="full_dataset_config.csv"):
    # label_file_base, label_file_name = os.path.splitext(label_file)
    # compressed_label_file = label_file_name + "_compressed.h5"
    label_file_base, label_file_name = os.path.split(label_file)
    compressed_label_file = "compressed_" + label_file_name
    abs_compressed_label_file = os.path.join(os.path.abspath(label_file_base), compressed_label_file)
    # config_file_name = config_file_name

    embedding_file = os.path.abspath(embedding_file)
    label_file = os.path.abspath(label_file)
    output_dir = os.path.abspath(output_dir)

    full_dataset_info = []
    if n_classes == 4:
        call_type_counts = {call_type: 0 for call_type in FOUR_CALL_TYPES}
    elif n_classes == 2:
        call_type_counts = {call_type: 0 for call_type in TWO_CALL_TYPES}

    # laod embeddings
    embeddings_file = h5py.File(embedding_file) 
    embeddings_dataset = embeddings_file["./tensor_dataset"]
    # laod labels
    # labels_df = pd.read_csv(label_file)
    labels_file = h5py.File(label_file, 'r')
    labels_dataset = labels_file["./tensor_dataset"]
    labels_df = pd.DataFrame(labels_dataset[:], columns=['foodCall', 'pleasureCall', 'distressCall', 'fearTrill', 'other'])

    # processing labels
    # drop fear trill
    labels_df['other'] = labels_df['other'] | labels_df['fearTrill']
    labels_df.drop(columns=['fearTrill'], inplace=True)
    # drop multi-hot
    one_hot_count = labels_df.sum(axis=1)
    labels_df.loc[one_hot_count > 1, ['foodCall', 'pleasureCall', 'distressCall']] = 0
    labels_df.loc[one_hot_count > 1, 'other'] = 1
    # n_classes 2 or 4
    if n_classes == 4:
        labels_df = labels_df[['foodCall', 'pleasureCall', 'distressCall', 'other']]
    elif n_classes == 2:
        # drop pleasure call, distress call
        labels_df['other'] = labels_df['other'] | labels_df['distressCall'] | labels_df['pleasureCall']
        labels_df.drop(columns=['distressCall'], inplace=True)
        labels_df.drop(columns=['pleasureCall'], inplace=True)
        labels_df = labels_df[['foodCall', 'other']]
    
    start_sample = 0
    start_embedding = 0
    total_compressed_labels = []
    print(f"len(embeddings_dataset): ", len(embeddings_dataset))
    while start_embedding < len(embeddings_dataset):
        end_embedding = start_embedding + 1
        
        start_sample = start_embedding * TRAIN_SAMPLES_PER_GROUP + 1
        end_sample = end_embedding * TRAIN_SAMPLES_PER_GROUP
        
        # compress label
        group_labels = labels_df.iloc[start_sample : end_sample]
        arrary_group_labels = group_labels.to_numpy(dtype=np.int32)
        group_mode = mode(arrary_group_labels, axis=0, keepdims=True).mode[0]
        # if other
        call_type = get_call_type(group_mode)
        if (call_type == "other" and n_classes == 4):
            total_compressed_labels.append([0,0,0,1])
        elif (call_type == "other" and n_classes == 2):
            total_compressed_labels.append([0,1])
        else:
            total_compressed_labels.append(group_mode)

        # signal
        timestamp = format_timestamp(start_sample / SAMPLE_RATE)
        print(f"timestamp: {timestamp}, call_type: {call_type}")        
        full_dataset_info.append({
            "call_type": call_type, 
            "embedding_index": start_embedding,
            "embedding_path": embedding_file,
            # "label_path": compressed_label_file,
            "label_path": abs_compressed_label_file,
            # "features_file": origin_features_folder + os.path.join(dataset_dir, new_dataset_file + ".h5"), 
            "subset": "train"
            })
        call_type_counts[call_type] += 1

        start_embedding = end_embedding

    if os.path.exists(abs_compressed_label_file):
        os.remove(abs_compressed_label_file)
    with h5py.File(abs_compressed_label_file, "w") as hdf:
        hdf.create_dataset("tensor_dataset", data=total_compressed_labels)

    full_dataset_df = pd.DataFrame(full_dataset_info)
    full_dataset_df.to_csv(os.path.join(output_dir, config_file_name), index=False)

    print("Dataset size:", len(full_dataset_info))
    for call_type, count in call_type_counts.items():
        print(f"{call_type}: {count}")

def csv_to_hdf(csv_file, hdf_file):
    labels_df = pd.read_csv(csv_file, header=0)
    dataset = labels_df.iloc[0 : -1]
    dataset_labels = dataset.to_numpy(dtype=np.int32)

    with h5py.File(hdf_file, "w") as hdf:
        hdf.create_dataset("tensor_dataset", data=dataset_labels)

    print(f"CSV file '{csv_file}' has been converted to HDF5 file '{hdf_file}'.")

def get_dataset_info(name, obj, datasets):
    if isinstance(obj, h5py.Dataset):
        # print(f"Group: {name}")
        datasets.append({
            "dataset": obj.name,
            "age": int(obj.attrs["age"][0]),
            "recording_number": int(obj.attrs["recording_number"][0]),
        })

def create_val_dataset(origin_features_folder, label_file, output_dir, prefix, config_file_name="val_dataset_config.csv"):    
    # convert to abs path
    origin_features_folder = os.path.abspath(origin_features_folder)
    label_file = os.path.abspath(label_file)
    output_dir = os.path.abspath(output_dir)

    full_dataset_info = []
    datasets = []
    with h5py.File(label_file, "r") as hdf:
        hdf.visititems(lambda name, obj: get_dataset_info(name, obj, datasets))
    # print(datasets)
    
    for item in datasets:
        dataset_path = item["dataset"]
        dataset_dir, dataset_file = os.path.split(dataset_path)
        new_dataset_file = prefix + dataset_file

        full_dataset_info.append({
            # "features_file": origin_features_folder + item["dataset"] + ".h5", 
            "features_file": origin_features_folder + os.path.join(dataset_dir, new_dataset_file + ".h5"), 
            "labels_file": label_file,
            "labels_dataset": item["dataset"],
            "subset": "val",
            "recording_number": item["recording_number"],
            "age": item["age"],
            })

    full_dataset_df = pd.DataFrame(full_dataset_info)
    full_dataset_df.to_csv(os.path.join(output_dir, config_file_name), index=False)

if __name__ == "__main__":
    # csv to hdf
    # csv_file = "./dataset/train_data/full_label.csv" 
    # hdf_file = "./dataset/train_data/full_label.h5"
    # csv_to_hdf(csv_file, hdf_file)

    save_dir = "./dataset/train_data/bandPassedDatas"
    origin_features_file = f"{save_dir}/bandPassed_fullSignal.h5"
    origin_label_file = "./dataset/train_data/full_label.h5"
    create_train_dataset(
        origin_features_file, origin_label_file, save_dir,
        n_classes=2,
        config_file_name="bandPassed_full_dataset_config.csv",
    )

    # save_dir = "./dataset/val_data"
    # origin_features_folder = f"{save_dir}/bandPassedDatas"
    # label_file = f"{save_dir}/val_data_labels.h5"
    # prefix = "bandPassed_"
    # create_val_dataset(
    #     origin_features_folder, label_file, save_dir, prefix,
    #     # n_classes=4,
    #     config_file_name="bandPassed_val_dataset_config.csv",
    # )
