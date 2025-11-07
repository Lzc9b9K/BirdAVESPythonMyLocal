# Environment
### Environment Info
```
conda env export --from-history
name: aves
channels:
  - defaults
dependencies:
  - cudnn
  - cudatoolkit=11.3
  - python=3.8
  - torchaudio==2.4.1
  - torchvision==0.19.1
  - conda-forge::ffmpeg
```
### Create Environment
```
conda env create -f environment.yml
```
### Use Conda
```
conda activate aves
```

### Trained Model Save Path
```
# rule
filename = scripts_train/train_xxxxxx.sh
model_save_path = logger_save_dir/logger_project
# e.g. 
filename = scripts_train/train_250219.sh
model_save_path = 2025_trained_model/svm_3_layer_model_0220
```

--------

# Prediction
```
# create BirdAVES features
script_create_aves_features_file.py

# predict the labels using SVM
script_features_svm_network_four_create_label.py
```