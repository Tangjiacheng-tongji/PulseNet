import numpy as np
import os
import torch
import pandas as pd

save_root="./checkpoint"

targets = ["rnn"]
meme_sizes={"rnn":[(50,1)]}
target_channels={"rnn":np.arange(64)}
target_stride={"rnn":(10,1)}

capacity = 3
keep_channel =  25

batch_size = 32
extract_iter = 32
prune_iter = 32
p = 0.2

def normalize(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    return (data - mean[:, None]) / std[:, None]

def smooth(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), 1, data)

def load_ucr_dataset(dataset_dir):
    data = {}
    for dataset_name in os.listdir(dataset_dir):
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.isdir(dataset_path):
            train_file = test_file = None
            for filename in os.listdir(dataset_path):
                if filename.endswith('_TRAIN.tsv'):
                    train_file = os.path.join(dataset_path, filename)
                    sep = '\t'
                elif filename.endswith('_TRAIN.csv'):
                    train_file = os.path.join(dataset_path, filename)
                    sep = ','
                elif filename.endswith('_TEST.tsv'):
                    test_file = os.path.join(dataset_path, filename)
                elif filename.endswith('_TEST.csv'):
                    test_file = os.path.join(dataset_path, filename)
            if train_file and test_file:
                train_data = pd.read_csv(train_file, sep=sep, header=None)
                test_data = pd.read_csv(test_file, sep=sep, header=None)

                train_data = train_data.fillna(value=0)
                test_data = test_data.fillna(value=0)

                train_labels = torch.tensor(train_data.iloc[:, 0].values)
                train_labels = train_labels - train_labels.min()
                train_values = train_data.iloc[:, 1:].values
                train_values = torch.tensor(((train_values)))
                train_values = train_values.unsqueeze(-1).float()

                test_labels = torch.tensor(test_data.iloc[:, 0].values)
                test_labels = test_labels - test_labels.min()
                test_values = test_data.iloc[:, 1:].values
                test_values = torch.tensor(((test_values)))
                test_values = test_values.unsqueeze(-1).float()

                data[dataset_name] = {
                    'train_data': train_values,
                    'train_labels': train_labels,
                    'test_data': test_values,
                    'test_labels': test_labels
                }
    return data

log='./log.txt'

num_channels = 5
num_labels = 3

momentum = 0.2
origin_interval = 2
