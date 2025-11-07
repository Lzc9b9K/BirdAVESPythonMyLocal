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
# SEGMENT_DURATION_S = 16000 # 1s
SEGMENT_DURATION_S = 16000 # 2s
SEGMENT_DURATION_GROUP_NUMS = 49 # 50 * 320 = 16000
SAMPLES_PER_GROUP = 326 # 49 * 326 = 15974 - 16000 =  -26
# SEGMENT_DURATION_S = 32000 # 2s
# SEGMENT_DURATION_GROUP_NUMS = 99 # 50 * 320 = 16000
# SAMPLES_PER_GROUP = 323 # 49 * 326 = 15974 - 16000 =  -26
# SEGMENT_DURATION_S = 48000 # 3s
# SEGMENT_DURATION_GROUP_NUMS = 149
# SAMPLES_PER_GROUP = 322
# SEGMENT_DURATION_S = 64000 # 4s
# SEGMENT_DURATION_GROUP_NUMS = 199
# SAMPLES_PER_GROUP = 321
# SEGMENT_DURATION_S = 80000 # 5s
# SEGMENT_DURATION_GROUP_NUMS = 249
# SAMPLES_PER_GROUP = 321

CALL_TYPES = [
    "foodCall", # Food call
    "pleasureCall", # Pleasuse call
    "distressCall", # Distress call
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
    if row['distressCall'] >= 1:
        return "distressCall"
    elif row['pleasureCall'] >= 1:
        return "pleasureCall"
    # elif row['fearTrill'] >= 1:
    #     # return "fearTrill"
    #     return "Other"
    elif row['foodCall'] >= 1:
        return "foodCall"
    elif row['other'] >= 1:
        return "other"
    else:
        return None

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(td.microseconds / 1000)
    return f"{hours:02}_{minutes:02}_{seconds:02}_{milliseconds:03}"

def split_audio(audio_file, label_file, output_dir, label_dir):
    full_dataset_info = []
    call_type_counts = {call_type: 0 for call_type in CALL_TYPES}

    # audio = AudioSegment.from_wav(audio_file)
    sr, audio = wavfile.read(audio_file)
    labels_df = pd.read_csv(label_file)

    labels_df['other'] = labels_df['other'] | labels_df['fearTrill']
    labels_df.drop(columns=['fearTrill'], inplace=True)
    one_hot_count = labels_df.sum(axis=1)
    # labels_df.loc[one_hot_count != 1, ['foodCall', 'pleasureCall', 'distressCall']] = 0
    labels_df.loc[one_hot_count > 1, 'other'] = 1
    labels_df = labels_df[['foodCall', 'pleasureCall', 'distressCall', 'other']]
    
    start_sample = 0
    while start_sample < len(audio):
        end_sample = start_sample + SEGMENT_DURATION_S
        segment = audio[start_sample:end_sample]

        labels = labels_df.iloc[start_sample:end_sample].sum()
        call_type = get_call_type(labels)
        # print(start_sample)
        if call_type:
            # signal
            timestamp = format_timestamp(start_sample / SAMPLE_RATE)
            # print("start_sample", start_sample)
            print(f"timestamp: {timestamp}, call_type: {call_type}")
            # filename = f"{call_type}_{timestamp}.wav"
            signal_filename = f"{timestamp}.wav"
            
            # compress labels
            compressed_labels = []
            for i in range(SEGMENT_DURATION_GROUP_NUMS):
                duration_labels = labels_df[start_sample:end_sample]
                group = duration_labels.iloc[i * SAMPLES_PER_GROUP:(i + 1) * SAMPLES_PER_GROUP]
                # print(f"group : {group}")
                # print(f"group sum: {group.sum()}")
                most_common_label = group.sum().idxmax()
                compressed_row = [0,0,0,0]
                compressed_row[labels_df.columns.get_loc(most_common_label)] = 1
                compressed_labels.append(compressed_row)
            compressed_df = pd.DataFrame(compressed_labels, columns=['foodCall', 'pleasureCall', 'distressCall', 'other'])
            compressed_df.to_csv(os.path.join(output_dir, f"{timestamp}.csv"), index=False)
            # save signal
            wavfile.write(os.path.join(output_dir, signal_filename), SAMPLE_RATE, segment)
            # save labels

            full_dataset_info.append({"filename": signal_filename, "call_type": call_type})
            call_type_counts[call_type] += 1
            start_sample = end_sample
        else:
            start_sample = start_sample + 1

    # dataset_df = pd.DataFrame(dataset_info, columns=['filepath', 'call_type'])
    full_dataset_df = pd.DataFrame(full_dataset_info)
    full_dataset_df.to_csv(os.path.join(label_dir, 'full_dataset.csv'), index=False)

    print("Dataset size:", len(full_dataset_info))
    # print("Call types:", dataset_df['call_type'].unique())
    for call_type, count in call_type_counts.items():
        print(f"{call_type}: {count}")

def save_train_and_test_dataset(fulldataset_file, all_dir, train_output_dir, test_output_dir, label_dir):
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
            info['labelpath'] = os.path.join(train_output_dir, f"{info['filename'][:-4]}.csv")
            train_info.append(info)
        
        for _, row in test_data.iterrows():
            info = row.to_dict()
            info['subset'] = 'test'
            info['filepath'] = os.path.join(test_output_dir, info['filename'])
            info['labelpath'] = os.path.join(test_output_dir, f"{info['filename'][:-4]}.csv")
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
        label_src_path = os.path.join(all_dir, f"{info['filename'][:-4]}.csv")
        label_dst_path = os.path.join(train_output_dir, f"{info['filename'][:-4]}.csv")
        shutil.copyfile(label_src_path, label_dst_path)
    for info in test_info:
        src_path = os.path.join(all_dir, info['filename'])
        dst_path = os.path.join(test_output_dir, info['filename'])
        shutil.copyfile(src_path, dst_path)
        label_src_path = os.path.join(all_dir, f"{info['filename'][:-4]}.csv")
        label_dst_path = os.path.join(test_output_dir, f"{info['filename'][:-4]}.csv")
        shutil.copyfile(label_src_path, label_dst_path)

    train_df = pd.DataFrame(train_info)
    train_df.to_csv(os.path.join(label_dir, 'train_dataset.csv'), index=False)
    test_df = pd.DataFrame(test_info)
    test_df.to_csv(os.path.join(label_dir, 'test_dataset.csv'), index=False)

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
    save_dir = "./dataset/20mins_4s_compressed_dataset"

    origin_audio_file = "./dataset/total/total_signal.wav"
    origin_label_file = "./dataset/total/total_label.csv"
    # label_dir = "./dataset/total"

    fulldataset_file = f"{save_dir}/full_dataset.csv"
    full_dir = f"{save_dir}/full"
    train_dir = f"{save_dir}/train"
    test_dir = f"{save_dir}/test"
    
    # os.makedirs(label_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    split_audio(
        origin_audio_file, origin_label_file, 
        full_dir, save_dir)

    # save_train_and_test_dataset(fulldataset_file, full_dir, train_dir, test_dir, save_dir)

    # train_dir = "./dataset/train/0_train_dataset.csv"
    # test_dir = "./dataset/test/0_test_dataset.csv"
    # show_dataset_count(train_dir, test_dir)
