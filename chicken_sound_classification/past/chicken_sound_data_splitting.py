import os
import shutil
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import soundfile as sf
from datetime import timedelta
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

SAMPLE_RATE = 16000
# DURATION = 16 * 60 # 16 mins
SEGMENT_DURATION_S = 16000 # 1s
CALL_TYPES = [
    "Fo", # Food call
    "Pl", # Pleasuse call
    "Di", # Distress call
    "FoPl", # Food call + Pleasure call 
    "FoDi", # Food call + Distress call
    "Fe", # Fear trill
    "Other", # Other
]

def get_call_type(row):
    if row['foodCall'] >= 1 and row['pleasureCall'] >= 1:
        return "FoPl"
        # return "Other"
    elif row['foodCall'] >= 1 and row['distressCall'] >= 1:
        return "FoDi"
        # return "Other"
    if row['distressCall'] >= 1:
        return "Di"
    elif row['pleasureCall'] >= 1:
        return "Pl"
    elif row['fearTrill'] >= 1:
        return "Fe"
        # return "Other"
    elif row['foodCall'] >= 1:
        return "Fo"
    elif row['other'] >= 1:
        return "Other"
    else:
        return None

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"

def split_audio(audio_file, label_file, output_dir):
    full_dataset_info = []
    call_type_counts = {call_type: 0 for call_type in CALL_TYPES}

    # audio = AudioSegment.from_wav(audio_file)
    sr, audio = wavfile.read(audio_file)
    labels_df = pd.read_csv(label_file)

    start_sample = 0
    while start_sample < len(audio):
        end_sample = start_sample + SEGMENT_DURATION_S
        segment = audio[start_sample:end_sample]

        labels = labels_df.iloc[start_sample:end_sample].sum()
        call_type = get_call_type(labels)
        # print(start_sample)
        if call_type:
            timestamp = format_timestamp(start_sample / SAMPLE_RATE)
            # print("start_sample", start_sample)
            # print("timestamp", timestamp)

            filename = f"{call_type}_{timestamp}.wav"

            wavfile.write(os.path.join(output_dir, filename), SAMPLE_RATE, segment)
    
            full_dataset_info.append({"filename": filename, "call_type": call_type})
            call_type_counts[call_type] += 1
            start_sample = end_sample
        else:
            start_sample = start_sample + 1

    # dataset_df = pd.DataFrame(dataset_info, columns=['filepath', 'call_type'])
    full_dataset_df = pd.DataFrame(full_dataset_info)
    full_dataset_df.to_csv(os.path.join(output_dir, 'full_dataset.csv'), index=False)

    print("Dataset size:", len(full_dataset_info))
    # print("Call types:", dataset_df['call_type'].unique())
    for call_type, count in call_type_counts.items():
        print(f"{call_type}: {count}")

def save_train_and_test_dataset(fulldataset_file, all_dir, train_output_dir, test_output_dir):
    train_info = []
    test_info = []

    fulldataset = pd.read_csv(fulldataset_file)

    for call_type in CALL_TYPES:
        # call_type_data = [info for info in fulldataset.columns if info['call_type'] == call_type]
        call_type_data = fulldataset[fulldataset['call_type'] == call_type]
        # print(call_type_data)
        train_data, test_data = train_test_split(call_type_data, train_size=0.7, test_size=0.3)
        # print(train_data)

        # train_info.extend(train_data.to_dict(orient='records'))
        # test_info.extend(test_data.to_dict(orient='records'))
        for _, row in train_data.iterrows():
            info = row.to_dict()
            info['subset'] = 'train'
            info['filepath'] = os.path.join(train_output_dir, info['filename'])
            train_info.append(info)
        
        for _, row in test_data.iterrows():
            info = row.to_dict()
            info['subset'] = 'test'
            info['filepath'] = os.path.join(test_output_dir, info['filename'])
            test_info.append(info)

        # for _, row in train_data.iterrows():
        #     info = row.to_dict()
        #     info['subset'] = 'val'
        #     info['filepath'] = os.path.join(train_output_dir, info['filename'])
        #     train_info.append(info)
    
    # print(train_info)
    for info in train_info:
        # print(info)
        src_path = os.path.join(all_dir, info['filename'])
        dst_path = os.path.join(train_output_dir, info['filename'])
        shutil.copyfile(src_path, dst_path)
    for info in test_info:
        src_path = os.path.join(all_dir, info['filename'])
        dst_path = os.path.join(test_output_dir, info['filename'])
        shutil.copyfile(src_path, dst_path)

    train_df = pd.DataFrame(train_info)
    train_df.to_csv(os.path.join(train_dir, '0_train_dataset.csv'), index=False)
    test_df = pd.DataFrame(test_info)
    test_df.to_csv(os.path.join(test_dir, '0_test_dataset.csv'), index=False)

def show_dataset_count(train_file, test_file):

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    plt.figure(figsize=(12, 6))

    # plt.subplot(1, 2, 1)
    sns.countplot(data=train_df, x='call_type')
    sns.countplot(data=test_df, x='call_type')
    plt.title('Train Data Distribution')
    plt.xlabel('Call Type')
    plt.ylabel('Count')
    # plt.subplot(1, 2, 2)
    # plt.title('Test Data Distribution')
    # plt.xlabel('Call Type')
    # plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    audio_file = "./dataset/highpass_150_chicken_sound.wav"
    label_file = "./dataset/trainLabel.csv"
    output_dir = "./dataset"

    fulldataset_file = "./dataset/all/full_dataset.csv"
    all_dir = "./dataset/all"
    train_dir = "./dataset/train"
    test_dir = "./dataset/test"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # split_audio(audio_file, label_file, all_dir)

    # check valSignal
    val_audio_file = "./dataset/00_original/valSignal.wav"
    val_label_file = "./dataset/00_original/valLabel.csv"
    val_output_dir = "./dataset/val/"
    # os.makedirs(val_audio_file, exist_ok=True)
    # os.makedirs(val_label_file, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    split_audio(val_audio_file, val_label_file, val_output_dir)

    # save_train_and_test_dataset(fulldataset_file, all_dir, train_dir, test_dir)

    # train_dir = "./dataset/train/0_train_dataset.csv"
    # test_dir = "./dataset/test/0_test_dataset.csv"
    # show_dataset_count(train_dir, test_dir)
