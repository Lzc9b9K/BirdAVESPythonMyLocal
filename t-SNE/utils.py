import torchaudio
def split_audio(input_file, output_file1, output_file2, split_time, sample_rate=16000):
  
    waveform, sr = torchaudio.load(input_file)

    if sr != sample_rate:
        raise ValueError(f"Sample rate of the file ({sr}) does not match the expected sample rate ({sample_rate}).")

    split_sample = int(split_time * sample_rate)

    waveform1 = waveform[:, :split_sample]
    waveform2 = waveform[:, split_sample:]
 
    torchaudio.save(output_file1, waveform1, sample_rate)
    torchaudio.save(output_file2, waveform2, sample_rate)

input_file = "./audios/chicken_sound.wav"

output_file1 = "./audios/chicken_sound_10.wav"
output_file2 = "./audios/chicken_sound_6.wav"

split_time = 10 * 60  # 10 mins

split_audio(input_file, output_file1, output_file2, split_time)