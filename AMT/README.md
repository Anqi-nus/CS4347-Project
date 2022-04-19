# CS4347-Project

Run `pip3 install -r requirements.txt` to set up environment. 

## AMT
### Dataset Preparation
Install [MIR-ST500 dataset](https://nusu-my.sharepoint.com/personal/e0740922_u_nus_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe0740922%5Fu%5Fnus%5Fedu%2FDocuments%2FCourses%2F2022%20春%20%2D%20Sound%20and%20Music%20Computing%2FDatasets%2FMIR%2DST500&ga=1) here. 
Split the dataset into test, train and valid. 
Run `python3 dataset.py --data_dir ./data --annotation_path ./data/annotations.json --save_dataset_dir ./data/`.

### Training
Run `python3 main.py`.
The default baseNN model will be used for training. 
If want to use a modified CNN, change the corresponding `baseNN` in `main.py` to `ModifiedCNN`, and re-run the command. 

After training is done, run `python3 eval.py` to get the best model id. 

### Inference
To run inference for a given audio file, run `python3 inference.py --model_dir ./path/to/model --music_path ./path/to/music`.

To run inference for a given directory of audio file, run `python3 inference.py --model_dir ./path/to/model --music_path ./path/to/music/directory`. 
All music files in the directory will be predicted. 

The output will be a MIDI file for each input audio, and a collated *./prediction.json* file consist of [onset, offset, pitch information] for all the input audio. 

