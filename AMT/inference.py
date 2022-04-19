import torch
from torch.utils.data import DataLoader

import os
import json
import argparse
import mido
from tqdm import tqdm
from pathlib import Path
from main import AST_Model
from dataset import OneSong

import warnings
warnings.filterwarnings('ignore')

def notes2mid(notes):
    print(notes)
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    mid.ticks_per_beat = 480
    new_tempo = mido.bpm2tempo(120.0)

    track.append(mido.MetaMessage('set_tempo', tempo=new_tempo))
    track.append(mido.Message('program_change', program=0, time=0))

    cur_total_tick = 0

    for note in notes:
        if note[2] == 0:
            continue
        note[2] = int(round(note[2]))
        ticks_since_previous_onset = int(mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo))
        ticks_current_note = int(mido.second2tick(note[1] - 0.0001, ticks_per_beat=480, tempo=new_tempo))
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick

        track.append(mido.Message('note_on', note=note[2], velocity=100, time=note_on_length))
        
        track.append(mido.Message('note_off', note=note[2], velocity=100, time=note_off_length))
        cur_total_tick = cur_total_tick + note_on_length + note_off_length

    return mid
    

def convert_to_midi(predicted_result, song_id, output_path):
    to_convert = predicted_result[song_id]
    mid = notes2mid(to_convert)
    mid.save(output_path)


def predict_one_song(model, input_path, song_id, results, song_name, onset_thres, offset_thres):
    test_dataset = OneSong(input_path, song_id)
    test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
    results = model.predict(test_loader, results=results, onset_thres=onset_thres, offset_thres=offset_thres)

    input_folder = input_path.split("/")[1]
    output_name = song_name + ".mid"
    output_path = os.path.join(input_folder, output_name)
    convert_to_midi(results, song_id, output_path)
    return results

def predict_whole_dir(model, test_dir, results, output_json_path, onset_thres, offset_thres):
    results = {}

    for song_dir in tqdm(os.listdir(test_dir)):
        if song_dir == ".DS_Store":
            continue
        input_path = os.path.join(test_dir, song_dir)
        song_name = song_dir.split(".")[0]
        results = predict_one_song(model, input_path, song_dir, results, song_name, onset_thres, offset_thres)

    with open(output_json_path, 'w') as f:
        output_string = json.dumps(results)
        f.write(output_string)
    return results
    
def make_predictions(music_path, output_path, model, onset_thres, offset_thres, song_id='1'):
    results = {}
    if os.path.isfile(music_path):
        results = predict_one_song(model, music_path, song_id, results, output_path=output_path, onset_thres=float(onset_thres), offset_thres=float(offset_thres))
    elif os.path.isdir(music_path):
        results = predict_whole_dir(model, music_path, results, output_json_path=output_path, onset_thres=float(onset_thres), offset_thres=float(offset_thres))
    else:
        print ("\"input\" argument is not valid")
    return results


if __name__ == '__main__':
    """
    This script performs inference using the trained singing transcription model in main.py.
    
    Sample usage:
    python inference.py --model_dir ./models/model_10 --music_path ./music/hey.mp3
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--music_path", default='./music', help="path to the input audio/folder")
    parser.add_argument('--output_path', default='./predictions.json', help="path to the output prediction json")
    parser.add_argument('--model_dir', default='./models/model_10', help='path to the trained model')
    parser.add_argument("--onset_thres", default=0.5, help="onset threshold")
    parser.add_argument("--offset_thres", default=0.8, help="offset threshold")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    model = AST_Model(device, args.model_dir)
    
    make_predictions(args.music_path, args.output_path, model, args.onset_thres, args.offset_thres)

    
