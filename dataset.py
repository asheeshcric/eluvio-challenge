# Using pytorch to implement an LSTM model on the text data, to predict whether the news is popular or not based on its title.
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class NewsData:
    
    def __init__(self, data_path, votes_threshold, val_pct=0.2):
        self.samples = []
        self._process_data()
        self.votes_threshold = votes_threshold
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Return features and label for the sample
        pass
    
    def _get_class(self, up_votes):
        # Convert up_votes into one of the two classes: Popular or Not Popular
        return 1 if up_votes >= self.votes_threshold else 0
    
    def _process_data(self):
        # Read csv as pandas dataframe
        dataset = pd.read_csv(self.data_path)
        # Drop down irrelevant features (Only "title" and "up_votes" are selected)
        dataset = dataset[['title', 'up_votes']]
        dataset.loc[dataset['up_votes'] >= self.votes_threshold, 'up_votes'] = 1
        dataset.loc[dataset['up_votes'] < self.votes_threshold, 'up_votes'] = 0
        
        