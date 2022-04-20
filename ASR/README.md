# CS4347-Project

## ASR
The automatic lyric transcription is completed with the speechbrain toolkits

### Model Training
To train the model, go to the path `PathToSpeechbrain/recipes/LibriSpeech/ASR/CTC`


Put the data folder under the same diroctory.


Run `python train_with_wav2vec.py hparams/train_with_wav2vec.yaml` to get the model

### Inference

Run `python inference.py hparams/train_with_wav2vec.yaml -m datapath.wav -o prediction.txt`

To predict the lyrics in `datapath.wav` and the result will be saved in `prediction.txt` file
