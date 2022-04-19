
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm

from model import BaseNN
from dataset import SingingDataset


FRAME_LENGTH = librosa.frames_to_time(1, sr=44100, hop_length=1024)

class AST_Model:
    '''
        This is main class for training model and making predictions.
    '''
    def __init__(self, device= "cuda:0", model_path=None):
        # Initialize model
        self.device = device 
        self.model = BaseNN().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args, learning_params):
        # Set paths
        trainset_path = args.train_dataset_path
        validset_path = args.valid_dataset_path
        save_model_dir = args.save_model_dir
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_params['lr'])
        onset_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0,], device=device))
        offset_criterion = nn.BCEWithLogitsLoss()
        octave_criterion = nn.CrossEntropyLoss(ignore_index=100)
        pitch_criterion = nn.CrossEntropyLoss(ignore_index=100)

        print('Loading datasets...')
        with open(trainset_path, 'rb') as f:
            trainset = pickle.load(f)
        with open(validset_path, 'rb') as f:
            validset = pickle.load(f)

        trainset_loader = DataLoader(
            trainset,
            batch_size=learning_params['batch_size'],
            num_workers=0,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        validset_loader = DataLoader(
            validset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        # Start training
        print('Start training...')
        start_time = time.time()

        for epoch in range(1, learning_params['epoch']+1):
            self.model.train()
            total_training_loss = 0
            total_training_split_loss = np.zeros(4)
            best_model_id = -1
            min_valid_loss = 10000            

            for batch_idx, batch in enumerate(trainset_loader):
        
                input_tensor = batch[0].to(device)
                onset_prob = batch[1][:, 0].float().to(device)
                offset_prob = batch[1][:, 1].float().to(device)
                pitch_octave = batch[1][:, 2].long().to(device)
                pitch_class = batch[1][:, 3].long().to(device)
               
                optimizer.zero_grad()

                onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

                loss_onset = onset_criterion(onset_logits, onset_prob)
                loss_offset = offset_criterion(offset_logits, offset_prob)
                loss_octave = octave_criterion(pitch_octave_logits, pitch_octave)
                loss_pitch = pitch_criterion(pitch_class_logits, pitch_class)

                total_training_split_loss[0] += loss_onset.item() 
                total_training_split_loss[1] += loss_offset.item()
                total_training_split_loss[2] += loss_octave.item()
                total_training_split_loss[3] += loss_pitch.item()

                loss = loss_onset + loss_offset + loss_octave + loss_pitch
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()
                
                if batch_idx % 5000 == 0 and batch_idx != 0:
                    print(epoch, batch_idx, "time:", time.time()-start_time, "loss:", total_training_loss/(batch_idx+1))
                    print('Split Train Loss: Onset {:.4f} Offset {:.4f} Pitch Octave {:.4f} Pitch Class {:.4f}'.format(
                        total_training_split_loss[0]/(batch_idx+1),
                        total_training_split_loss[1]/(batch_idx+1),
                        total_training_split_loss[2]/(batch_idx+1),
                        total_training_split_loss[3]/(batch_idx+1)))

            if epoch % learning_params['valid_freq'] == 0:
                self.model.eval()
                with torch.no_grad():
                    total_valid_loss = 0
                    total_valid_split_loss = np.zeros(4)
                    for batch_idx, batch in enumerate(validset_loader):

                        input_tensor = batch[0].to(device)

                        onset_prob = batch[1][:, 0].float().to(device)
                        offset_prob = batch[1][:, 1].float().to(device)
                        pitch_octave = batch[1][:, 2].long().to(device)
                        pitch_class = batch[1][:, 3].long().to(device)

                        onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits = self.model(input_tensor)

                        loss_onset = onset_criterion(onset_logits, onset_prob)
                        loss_offset = offset_criterion(offset_logits, offset_prob)
                        loss_octave = octave_criterion(pitch_octave_logits, pitch_octave)
                        loss_pitch = pitch_criterion(pitch_class_logits, pitch_class)

                        total_valid_split_loss[0] += loss_onset.item() 
                        total_valid_split_loss[1] += loss_offset.item()
                        total_valid_split_loss[2] += loss_octave.item()
                        total_valid_split_loss[3] += loss_pitch.item()
                        
                        # Calculate loss
                        loss = loss_onset + loss_offset + loss_octave + loss_pitch
                        total_valid_loss += loss.item()

                # Save model
                if epoch % learning_params['save_freq'] == 0:
                    save_dict = self.model.state_dict()
                    target_model_path = save_model_dir + '/model_' + str(epoch)
                    torch.save(save_dict, target_model_path)

                # Epoch statistics
                print('| Epoch [{:4d}/{:4d}] Train Loss {:.4f} Valid Loss {:.4f} Time {:.1f}'.format(
                        epoch,
                        learning_params['epoch'],
                        total_training_loss/len(trainset_loader),
                        total_valid_loss/len(validset_loader),
                        time.time()-start_time))

                print('Split Valid Loss: Onset {:.4f} Offset {:.4f} Pitch Octave {:.4f} Pitch Class {:.4f}'.format(
                        total_valid_split_loss[0]/len(validset_loader),
                        total_valid_split_loss[1]/len(validset_loader),
                        total_valid_split_loss[2]/len(validset_loader),
                        total_valid_split_loss[3]/len(validset_loader)))
                
                if total_valid_loss/len(validset_loader) < min_valid_loss:
                    min_valid_loss = total_valid_loss/len(validset_loader)
                    best_model_id = epoch

        print('Training done in {:.1f} minutes.'.format((time.time()-start_time)/60))
        return best_model_id


    def parse_frame_info(self, frame_info, onset_thres, offset_thres):
        """Parse frame info [(onset_probs, offset_probs, note number)...] into desired label format."""

        result = []
        current_onset = None
        pitch_counter = []
        local_max_size = 3
        current_time = 0.0

        onset_seq = np.array([frame_info[i][0] for i in range(len(frame_info))])
        onset_seq_length = len(onset_seq)

        for i in range(len(frame_info)):

            current_time = FRAME_LENGTH*i
            info = frame_info[i]

            backward_frames = i - local_max_size
            if backward_frames < 0:
                backward_frames = 0

            forward_frames = i + local_max_size + 1
            if forward_frames > onset_seq_length - 1:
                forward_frames = onset_seq_length - 1

            # local max and more than threshold
            if info[0] >= onset_thres and onset_seq[i] == np.amax(onset_seq[backward_frames : forward_frames]):

                if current_onset is None:
                    current_onset = current_time   
                else:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = current_time
                    pitch_counter = []

            # if it is offset
            elif info[1] >= offset_thres:  
                if current_onset is not None:
                    if len(pitch_counter) > 0:
                        result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
                    current_onset = None
                    pitch_counter = []

            # If current_onset exist, add count for the pitch
            if current_onset is not None:
                final_pitch = int(info[2]* 12 + info[3])
                if info[2] != 4 and info[3] != 12:
                    pitch_counter.append(final_pitch)

        if current_onset is not None:
            if len(pitch_counter) > 0:
                result.append([current_onset, current_time, max(set(pitch_counter), key=pitch_counter.count) + 36])
            current_onset = None

        return result

    def predict(self, test_loader, results={}, onset_thres=0.1, offset_thres=0.5):
        """Predict results for a given test dataset."""

        # Start predicting
        self.model.eval()
        with torch.no_grad():
            song_frames_table = {}
            
            for batch_idx, batch in enumerate(tqdm(test_loader)):
                # Parse batch data
                input_tensor = batch[0].to(self.device)
                song_ids = batch[1]

                result_tuple = self.model(input_tensor)
                onset_logits = result_tuple[0]
                offset_logits = result_tuple[1]
                pitch_octave_logits = result_tuple[2]
                pitch_class_logits = result_tuple[3]

                onset_probs, offset_probs = torch.sigmoid(onset_logits).cpu(), torch.sigmoid(offset_logits).cpu()
                pitch_octave_logits, pitch_class_logits = pitch_octave_logits.cpu(), pitch_class_logits.cpu()

                # Collect frames for corresponding songs
                for bid, song_id in enumerate(song_ids):
                    frame_info = (onset_probs[bid], offset_probs[bid], torch.argmax(pitch_octave_logits[bid]).item()
                            , torch.argmax(pitch_class_logits[bid]).item())

                    song_frames_table.setdefault(song_id, [])
                    song_frames_table[song_id].append(frame_info)    

            # Parse frame info into output format for every song
            for song_id, frame_info in song_frames_table.items():
                results[song_id] = self.parse_frame_info(frame_info, onset_thres=onset_thres, offset_thres=offset_thres)
        
        return results


if __name__ == '__main__':
    """
    This script performs training and validation of the singing transcription model.
    
    Sample usage:
    python main.py --train_dataset_path ./data/train.pkl --valid_dataset_path ./data/valid.pkl --save_model_dir ./results
    or 
    python main.py (All parameters are defualt)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', default='./data/train.pkl', help='path to the train set')
    parser.add_argument('--valid_dataset_path', default='./data/valid.pkl', help='path to the valid set')
    parser.add_argument('--save_model_dir', default='./results', help='path to save the trained models')
    
    args = parser.parse_args()
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ast_model = AST_Model(device)

    # Set learning params
    learning_params = {
        'batch_size': 50,
        'epoch': 10,
        'lr': 1e-4,
        'valid_freq': 1,
        'save_freq': 1
    }
    # Train and Validation
    best_model_id = ast_model.fit(args, learning_params)
    print("Best Model ID: ", best_model_id)
    
    



    
