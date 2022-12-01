import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random
from tqdm import tqdm
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


class TranslateDataset(Dataset):
    def __init__(self, data_dirs, dict_list):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--states.json
        #     |--trial1
        #     |--...


        with open(dict_list) as json_file:
            self.dict_list = json.load(json_file)
            json_file.close()

        all_dirs = []
        for data_dir in data_dirs:
            all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        self.trials = []

        print('Start Loading Data!')
        for trial in tqdm(all_dirs):

            states_json = os.path.join(trial, 'states_detailed_sentence.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            

            trial_dict = {}
            trial_dict['img_path'] = os.path.join(trial, str(0) + '.png')
            trial_dict['sentence'] = states_dict[0]['sentence']
            trial_dict['detailed_sentence'] = states_dict[0]['detailed_sentence']
            for i in range(len(trial_dict['detailed_sentence'])):
                if 'milk_bottle' in trial_dict['detailed_sentence'][i]:
                    trial_dict['detailed_sentence'][i] = trial_dict['detailed_sentence'][i].replace('milk_bottle', 'milk bottle')
            trial_dict['detailed_sentence'] = ' <SEP> '.join(trial_dict['detailed_sentence'])
            trial_dict['detailed_sentence'] = '<SOS> ' + trial_dict['detailed_sentence'] + ' <EOS>'

            self.trials.append(trial_dict)

        _, self.preprocess = clip.load('RN50', device='cuda')
        print('Data Loaded!')

    def __len__(self):
        return len(self.trials)

    def __tokenize_detailed_sentence__(self, detailed_sentence):
        tokens = detailed_sentence.split(' ')
        for i in range(len(tokens)):
            tokens[i] = self.dict_list[tokens[i]]
        return tokens

    def __getitem__(self, index):

        img_file = self.trials[index]['img_path']
        img = Image.open(img_file)
        img = self.preprocess(img)

        sentence = self.trials[index]['sentence']
        sentence = clip.tokenize([sentence])[0]
        
        detailed_sentence = self.trials[index]['detailed_sentence']
        # print(detailed_sentence)
        detailed_sentence = torch.tensor(self.__tokenize_detailed_sentence__(detailed_sentence), dtype=torch.int64)
        # print(detailed_sentence)

        return img, sentence, detailed_sentence


def pad_collate_xy_lang(batch):
    (img, sentence, detailed_sentence) = zip(*batch)

    img = torch.stack(img)
    sentence = torch.stack(sentence)
    detailed_sentence = pad_sequence(detailed_sentence, batch_first=True, padding_value=0)

    return img, sentence, detailed_sentence


class Untokenizer:
    def __init__(self, dict_list):
        with open(dict_list) as json_file:
            self.dict_list = json.load(json_file)
            json_file.close()
        self.inv_dict_list = [''] * len(self.dict_list)
        for token in self.dict_list:
            idx = self.dict_list[token]
            self.inv_dict_list[idx] = token

    def untokenize(self, indices):
        # print(indices)
        tokens = []
        for index in indices:
            token = self.inv_dict_list[index]
            tokens.append(token)
        return ' '.join(tokens)



def get_voc(data_dirs):
    all_dirs = []
    for data_dir in data_dirs:
        all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

    dict_list = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<SEP>': 2,
        '<EOS>': 3,
    }
    for trial in tqdm(all_dirs):
        states_json = os.path.join(trial, 'states_detailed_sentence.json')
        with open(states_json) as json_file:
            states_dict = json.load(json_file)
            json_file.close()
            detailed_sentence = states_dict[0]['detailed_sentence']
            for sentence in detailed_sentence:
                if 'milk_bottle' in sentence:
                    sentence = sentence.replace("milk_bottle", "milk bottle")
                tokens = sentence.strip().split(' ')
                for token in tokens:
                    if token not in dict_list:
                        dict_list[token] = len(dict_list)

    print(dict_list)
    with open("dict_list.json", "w") as outfile:
        json.dump(dict_list, outfile)



if __name__ == '__main__':
    data_dirs = [
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_detailed_sentence',
        '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_val_detailed_sentence',
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_detailed_sentence',
        # '/data/Documents/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_val_detailed_sentence'
    ]
    dataset = TranslateDataset(data_dirs, 'dict_list.json')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                          shuffle=True, num_workers=2,
                                          collate_fn=pad_collate_xy_lang)
    i = 0
    for x, y, z in dataloader:
        print(y, z)
        i += 1
        if i == 5:
            exit()
    # get_voc(data_dirs)
